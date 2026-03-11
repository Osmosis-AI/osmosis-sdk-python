"""Project resolution, caching, and interactive selection for platform CLI."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import (
    Choice,
    Separator,
    confirm,
    is_interactive,
    select,
    text,
)
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    get_active_workspace,
    load_workspace_credentials,
    load_workspace_projects,
    save_workspace_projects,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    get_default_project,
    load_subscription_status,
    save_subscription_status,
)
from osmosis_ai.platform.cli.constants import (
    BACK,
    CACHE_TTL_SECONDS,
    CREATE,
    MSG_NOT_LOGGED_IN,
    PROJECT_NAME_MAX,
    PROJECT_NAME_RE,
    RESERVED_PROJECT_NAMES,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import WorkspaceCredentials


def validate_project_name(name: str) -> str | None:
    """Validate a project name against platform rules.

    Returns None if valid, or an error message string if invalid.
    """
    if not name:
        return "Project name is required."
    if len(name) > PROJECT_NAME_MAX:
        return f"Project name must be {PROJECT_NAME_MAX} characters or less."
    if name != name.lower():
        return "Project name must be lowercase."
    if not PROJECT_NAME_RE.match(name):
        return (
            "Project name must contain only lowercase letters, digits, and hyphens, "
            "and cannot start or end with a hyphen."
        )
    if name in RESERVED_PROJECT_NAMES:
        return f"'{name}' is a reserved name and cannot be used."
    return None


def _get_active_workspace_name() -> str:
    """Return the active workspace name, or raise if none is selected."""
    workspace_name = get_active_workspace()
    if workspace_name is None:
        raise CLIError(MSG_NOT_LOGGED_IN)
    return workspace_name


def _get_workspace_credentials(workspace_name: str) -> WorkspaceCredentials:
    """Load valid credentials for a specific workspace."""
    credentials = load_workspace_credentials(workspace_name)
    if credentials is None or credentials.is_expired():
        raise AuthenticationExpiredError(
            "No valid credentials found. Please run 'osmosis login' first."
        )

    return credentials


# ── Interactive selection ──────────────────────────────────────────


def select_project_interactive(
    ws_name: str,
    projects: list[dict[str, Any]] | None = None,
    current_project_id: str | None = None,
    allow_back: bool = False,
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

    Returns:
        Dict with 'id' and 'project_name', BACK if back selected,
        or None if skipped/cancelled.
    """
    if projects is None:
        try:
            projects = _refresh_projects(workspace_name=ws_name)
        except AuthenticationExpiredError:
            raise
        except Exception:
            projects = _get_cached_projects(workspace_name=ws_name, max_age=None)
            if projects:
                console.print(
                    "Warning: Failed to refresh projects, using cached data.",
                    style="yellow",
                )

    if not is_interactive():
        # Non-interactive: auto-select if exactly one, otherwise skip
        if len(projects) == 1:
            return projects[0]
        return None

    if not projects:
        console.print(f"No projects in '{ws_name}'.")
        return _prompt_create(ws_name)

    return _prompt_select(ws_name, projects, current_project_id, allow_back=allow_back)


def _prompt_create(ws_name: str) -> dict | None:
    """Prompt to create a new project."""
    while True:
        name = text(
            "Project name:",
            instruction="lowercase letters, digits, and hyphens only",
        )

        if name is None:
            return None

        error = validate_project_name(name)
        if error:
            console.print_error(error)
            continue

        break

    ok = confirm(f"Create project '{name}'?")
    if not ok:
        return None

    credentials = _get_workspace_credentials(ws_name)
    client = OsmosisClient()
    project = client.create_project(name, credentials=credentials)

    try:
        _refresh_projects(workspace_name=ws_name)
    except AuthenticationExpiredError:
        raise
    except Exception:
        pass

    console.print(f"Created project '{project.project_name}'.")
    return {"id": project.id, "project_name": project.project_name}


def _prompt_select(
    ws_name: str,
    projects: list[dict],
    current_project_id: str | None,
    allow_back: bool = False,
) -> dict | str | None:
    """Display project list with arrow-key selection and create option."""
    # Build choices for ALL projects (no DISPLAY_LIMIT cap)
    choices = []

    for p in projects:
        is_current = p.get("id") == current_project_id
        title = p["project_name"]
        if is_current:
            title += " (current)"
        choices.append(Choice(title, value=p))

    # Add separator + create option + optional back
    choices.append(Separator())
    choices.append(Choice("Create new project", value=CREATE))
    if allow_back:
        choices.append(Choice("Back", value=BACK))

    # Prompt user
    console.separator()
    result = select(
        "Select a project:",
        choices=choices,
    )

    if result is None:
        return None

    # Handle special options
    if result == BACK:
        return BACK
    if result == CREATE:
        return _prompt_create(ws_name)

    # Otherwise result is the project dict
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
        is_stale = refreshed_at is None or (time.time() - refreshed_at) > max_age
        if is_stale:
            try:
                return _refresh_projects(workspace_name=workspace_name)
            except AuthenticationExpiredError:
                raise
            except Exception:
                console.print_error(
                    "Warning: Could not refresh project list, using cached data."
                )
                return projects

    return projects


def _refresh_projects(*, workspace_name: str) -> list[dict]:
    """Refresh project list from platform and update cache."""
    credentials = _get_workspace_credentials(workspace_name)
    client = OsmosisClient()
    info = client.refresh_workspace_info(credentials=credentials)
    projects = info.get("projects", [])
    save_workspace_projects(workspace_name, projects)
    has_subscription = info.get("has_subscription")
    if has_subscription is not None:
        save_subscription_status(workspace_name, bool(has_subscription))
    return projects


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
    if refresh or not projects:
        projects = _refresh_projects(workspace_name=workspace_name)

    target = name_or_id.lower()
    for p in projects:
        if p.get("project_name", "").lower() == target or p.get("id") == name_or_id:
            return p

    if not refresh:
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
) -> tuple[str, WorkspaceCredentials]:
    """Check that user is authenticated for a workspace."""
    if workspace_name is None:
        workspace_name = _get_active_workspace_name()

    credentials = _get_workspace_credentials(workspace_name)

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
    try:
        _refresh_projects(
            workspace_name=workspace_name
        )  # Also updates subscription cache
    except AuthenticationExpiredError:
        raise
    except Exception:
        pass

    # Re-check after refresh (no TTL — we just refreshed)
    status = load_subscription_status(workspace_name)
    if status is not True:
        raise CLIError(
            "Your workspace requires an active subscription for this action.\n"
            f"  Upgrade at: {PLATFORM_URL}/{workspace_name}/settings/billing"
        )
