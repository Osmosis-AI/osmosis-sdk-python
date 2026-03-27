"""Project resolution, caching, and interactive selection for platform CLI."""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import (
    Choice,
    confirm,
    is_interactive,
    select_list,
    text,
)
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth import (
    PlatformAPIError,
    load_workspace_projects,
    save_workspace_projects,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    get_active_workspace_name,
    get_default_project,
    load_subscription_status,
    save_subscription_status,
)
from osmosis_ai.platform.cli.constants import (
    BACK,
    CACHE_TTL_SECONDS,
    CREATE,
    DEFAULT_VISIBLE_CHOICES,
    RESERVED_PROJECT_NAMES,
    validate_name,
)
from osmosis_ai.platform.cli.utils import require_credentials

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def validate_project_name(name: str) -> str | None:
    """Validate a project name against platform rules.

    Returns None if valid, or an error message string if invalid.
    """
    error = validate_name(name, label="Project name")
    if error:
        return error
    if name in RESERVED_PROJECT_NAMES:
        return f"'{name}' is a reserved name and cannot be used."
    return None


def _get_active_workspace_name() -> str:
    """Return the active workspace name, or raise if none is selected."""
    workspace_name = get_active_workspace_name()
    if workspace_name is None:
        raise CLIError(
            "No workspace selected. Run 'osmosis workspace' to select a workspace."
        )
    return workspace_name


# ── Interactive selection ──────────────────────────────────────────


def select_project_interactive(
    ws_name: str,
    projects: list[dict[str, Any]] | None = None,
    current_project_id: str | None = None,
    allow_back: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any] | str | None:
    """Interactively select or create a project.

    - 0 projects: prompt to create (TTY only)
    - 1 project: auto-select (works in any environment)
    - N projects: numbered list with search and create option (TTY only)

    Args:
        ws_name: Workspace name for display.
        projects: Pre-fetched project list, or None to fetch from API.
        current_project_id: ID of the current default (for "current" marker).
        allow_back: If True, show a "Back" option to return to previous step.
        workspace_id: Explicit workspace ID for API calls (used before workspace
            is persisted as active).

    Returns:
        Dict with 'id' and 'project_name', BACK if back selected,
        or None if skipped/cancelled.
    """
    if projects is None:
        try:
            with console.spinner("Loading projects..."):
                projects = _refresh_projects(
                    workspace_name=ws_name, workspace_id=workspace_id
                )
        except (PlatformAPIError, OSError):
            projects = _get_cached_projects(workspace_name=ws_name, max_age=None)
            if projects:
                console.print(
                    "Warning: Failed to refresh projects, using cached data.",
                    style="yellow",
                )
            else:
                raise CLIError(
                    "Could not load project list and no cached data is available.\n"
                    "  Please check your network connection and try again."
                ) from None

    if not is_interactive():
        # Non-interactive: auto-select if exactly one, otherwise skip
        if len(projects) == 1:
            return projects[0]
        return None

    if not projects:
        console.print(f"No projects in '{ws_name}'.")
        return _prompt_create(ws_name, workspace_id=workspace_id)

    return _prompt_select(
        ws_name,
        projects,
        current_project_id,
        allow_back=allow_back,
        workspace_id=workspace_id,
    )


def _prompt_create(ws_name: str, workspace_id: str | None = None) -> dict | None:
    """Prompt to create a new project."""
    name = text(
        "Project name:",
        instruction="lowercase letters, digits, and hyphens only",
        validate=lambda v: validate_project_name(v) or True,
    )

    if name is None:
        return None

    ok = confirm(f"Create project '{name}'?")
    if not ok:
        return None

    credentials = require_credentials()
    client = OsmosisClient()
    project = client.create_project(
        name, credentials=credentials, workspace_id=workspace_id
    )

    with contextlib.suppress(PlatformAPIError, OSError):
        _refresh_projects(workspace_name=ws_name, workspace_id=workspace_id)

    console.print(f"Created project '{project.project_name}'.")
    return {"id": project.id, "project_name": project.project_name}


def _prompt_select(
    ws_name: str,
    projects: list[dict],
    current_project_id: str | None,
    allow_back: bool = False,
    workspace_id: str | None = None,
) -> dict | str | None:
    """Display scrollable project list with pinned action items."""
    items = []
    default_value = None
    for p in projects:
        is_current = p.get("id") == current_project_id
        title = p["project_name"]
        if is_current:
            title += " (current)"
            default_value = p
        items.append(Choice(title, value=p))

    actions: list[Choice] = [Choice("Create new project", value=CREATE)]
    if allow_back:
        actions.append(Choice("Back", value=BACK))

    console.separator()
    result = select_list(
        "Choose a project",
        items=items,
        actions=actions,
        default=default_value,
        max_visible=DEFAULT_VISIBLE_CHOICES,
    )

    if result is None:
        return None
    if result == BACK:
        return BACK
    if result == CREATE:
        return _prompt_create(ws_name, workspace_id=workspace_id)

    return result


# ── Caching ────────────────────────────────────────────────────────


def _get_cached_projects(
    *,
    workspace_name: str,
    max_age: float | None = CACHE_TTL_SECONDS,
) -> list[dict]:
    """Load cached projects, auto-refreshing if stale."""
    projects, refreshed_at = load_workspace_projects(workspace_name)

    if max_age is not None and projects:
        is_stale = (
            not isinstance(refreshed_at, (int, float))
            or (time.time() - refreshed_at) > max_age
        )
        if is_stale:
            try:
                return _refresh_projects(workspace_name=workspace_name)
            except (PlatformAPIError, OSError):
                console.print_error(
                    "Warning: Could not refresh project list, using cached data."
                )
                return projects

    return projects


def _refresh_projects(
    *, workspace_name: str, workspace_id: str | None = None
) -> list[dict]:
    """Refresh project list from platform and update cache."""
    credentials = require_credentials()
    client = OsmosisClient()
    all_projects: list[dict] = []
    offset = 0
    page_size = 50

    while True:
        result = client.list_projects(
            limit=page_size,
            offset=offset,
            credentials=credentials,
            workspace_id=workspace_id,
        )
        all_projects.extend(p.to_dict() for p in result.projects)
        if result.next_offset is None:
            break
        offset = result.next_offset

    save_workspace_projects(workspace_name, all_projects)
    return all_projects


# ── Resolution (non-interactive, for commands like dataset) ────────


def _resolve_project(
    name_or_id: str | None,
    *,
    workspace_name: str,
    refresh: bool = False,
) -> dict:
    """Resolve a project by name or ID, using cache first.

    Priority: --project arg > $OSMOSIS_PROJECT > config.json default > error
    """
    if name_or_id is None:
        name_or_id = os.environ.get("OSMOSIS_PROJECT")

    if name_or_id is None:
        default = get_default_project(workspace_name)
        if default:
            name_or_id = default.get("project_name")

    if name_or_id is None:
        available = _get_cached_projects(workspace_name=workspace_name, max_age=None)
        hint = (
            "No project specified.\n\n"
            "Specify a project in one of these ways:\n"
            "  --project <name>                   Flag on the command\n"
            "  export OSMOSIS_PROJECT=<name>       Environment variable\n"
            "  osmosis workspace                   Set a default interactively"
        )
        if available:
            names = ", ".join(p["project_name"] for p in available)
            hint += f"\n\nAvailable projects: {names}"
        raise CLIError(hint)

    projects = _get_cached_projects(workspace_name=workspace_name)
    refreshed = False
    if refresh or not projects:
        projects = _refresh_projects(workspace_name=workspace_name)
        refreshed = True

    target = name_or_id.lower()
    for p in projects:
        if p.get("project_name", "").lower() == target or p.get("id") == name_or_id:
            return p

    if not refreshed:
        projects = _refresh_projects(workspace_name=workspace_name)
        for p in projects:
            if p.get("project_name", "").lower() == target or p.get("id") == name_or_id:
                return p

    raise CLIError(f"Project '{name_or_id}' not found in this workspace.")


def _resolve_project_id(project: str | None, *, workspace_name: str) -> str:
    """Get project ID from --project arg, env, or default."""
    proj = _resolve_project(project, workspace_name=workspace_name)
    return proj["id"]


def _require_auth(
    *,
    workspace_name: str | None = None,
) -> tuple[str, Credentials]:
    """Check that user is authenticated and has a workspace selected.

    Checks credentials first so that unauthenticated users see "Not logged in"
    instead of the misleading "No workspace selected".
    """
    credentials = require_credentials()
    if workspace_name is None:
        workspace_name = _get_active_workspace_name()
    return workspace_name, credentials


def _require_subscription(*, workspace_name: str) -> None:
    """Check that a workspace has an active subscription.

    Uses cached status with TTL. If the cache is expired, stale, or False,
    refreshes from the platform to avoid blocking users who just subscribed.
    """
    cached = load_subscription_status(workspace_name, max_age=CACHE_TTL_SECONDS)
    if cached is True:
        return

    # Cached status is False, None, or expired — refresh to get the latest
    refreshed = False
    with contextlib.suppress(PlatformAPIError, OSError):
        credentials = require_credentials()
        client = OsmosisClient()
        info = client.refresh_workspace_info(
            credentials=credentials, workspace_name=workspace_name
        )
        has_subscription = info.get("has_subscription")
        if has_subscription is not None:
            save_subscription_status(workspace_name, bool(has_subscription))
            refreshed = True

    # Re-check after refresh attempt
    status = load_subscription_status(workspace_name, max_age=CACHE_TTL_SECONDS)
    if status is True:
        return

    # If refresh failed and status is still unknown, don't block the user
    if not refreshed and status is None:
        return

    raise CLIError(
        "Your workspace requires an active subscription for this action.\n"
        f"  Upgrade at: {PLATFORM_URL}/{workspace_name}/settings/billing"
    )
