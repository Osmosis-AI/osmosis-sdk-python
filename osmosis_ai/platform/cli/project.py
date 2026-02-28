"""Project resolution, caching, and interactive selection for platform CLI."""

from __future__ import annotations

import contextlib
import os
import re
import time

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import Choice, Separator, is_interactive, select, text
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth import (
    PlatformAPIError,
    get_active_workspace,
    get_valid_credentials,
    load_workspace_projects,
    save_workspace_projects,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    get_default_project,
    load_subscription_status,
    save_subscription_status,
)

CACHE_TTL_SECONDS = 300

# Must match the frontend validation in osmosis-monolith:
# platform-app/src/constants/projects.ts + org-routes.ts
_PROJECT_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")
_PROJECT_NAME_MAX = 64
_RESERVED_PROJECT_NAMES = frozenset(
    {
        # Org-level route segments
        "projects",
        "data-sources",
        "tools",
        "reward-functions",
        "llm-judges",
        "rollout-servers",
        "settings",
        # System reserved
        "api",
        "admin",
        "new",
        "project",
    }
)


def validate_project_name(name: str) -> str | None:
    """Validate a project name against platform rules.

    Returns None if valid, or an error message string if invalid.
    """
    if not name:
        return "Project name is required."
    if len(name) > _PROJECT_NAME_MAX:
        return f"Project name must be {_PROJECT_NAME_MAX} characters or less."
    if name != name.lower():
        return "Project name must be lowercase."
    if not _PROJECT_NAME_RE.match(name):
        return (
            "Project name must contain only lowercase letters, digits, and hyphens, "
            "and cannot start or end with a hyphen."
        )
    if name in _RESERVED_PROJECT_NAMES:
        return f"'{name}' is a reserved name and cannot be used."
    return None


# ── Interactive selection ──────────────────────────────────────────


def select_project_interactive(
    ws_name: str,
    projects: list[dict] | None = None,
    current_project_id: str | None = None,
) -> dict | None:
    """Interactively select or create a project.

    - 0 projects: prompt to create (TTY only)
    - 1 project: auto-select (works in any environment)
    - N projects: numbered list with search and create option (TTY only)

    Args:
        ws_name: Workspace name for display.
        projects: Pre-fetched project list, or None to fetch from API.
        current_project_id: ID of the current default (for "current" marker).

    Returns:
        Dict with 'id' and 'project_name', or None if skipped/cancelled.
    """
    if projects is None:
        try:
            projects = _refresh_projects()
        except Exception:
            projects = _get_cached_projects(max_age=None)
            if projects:
                console.print_error(
                    "Warning: Could not reach platform, showing cached data."
                )

    if not is_interactive():
        # Non-interactive: auto-select if exactly one, otherwise skip
        if len(projects) == 1:
            return projects[0]
        return None

    if not projects:
        console.print(f"No projects in '{ws_name}'.")
        return _prompt_create()

    return _prompt_select(projects, current_project_id)


def _prompt_create() -> dict | None:
    """Prompt to create a new project."""

    def validate_input(value: str) -> bool | str:
        """Validator for questionary text prompt."""
        if not value:
            return "Project name is required."
        # Normalize to lowercase first
        normalized = value.lower()
        error = validate_project_name(normalized)
        if error:
            return error
        return True

    name = text(
        "Project name:",
        validate=validate_input,
        instruction="lowercase letters, digits, and hyphens only",
    )

    if name is None:
        return None

    # Normalize to lowercase (matching frontend behavior)
    name = name.lower()

    try:
        client = OsmosisClient()
        project = client.create_project(name)
    except PlatformAPIError as e:
        console.print_error(f"Failed to create project: {e}")
        return None

    with contextlib.suppress(Exception):
        _refresh_projects()

    console.print(f"Created project '{project.project_name}'.")
    return {"id": project.id, "project_name": project.project_name}


def _prompt_select(
    projects: list[dict],
    current_project_id: str | None,
) -> dict | None:
    """Display project list with arrow-key selection and create option."""
    # Build choices for ALL projects (no DISPLAY_LIMIT cap)
    choices = []

    for p in projects:
        is_current = p.get("id") == current_project_id
        title = p["project_name"]
        if is_current:
            title += " (current)"
        choices.append(Choice(title, value=p))

    # Add separator + create option
    choices.append(Separator())
    choices.append(Choice("Create new project", value="__create__"))

    # Prompt user
    result = select(
        "Select a project:",
        choices=choices,
    )

    if result is None:
        return None

    # Handle create option
    if result == "__create__":
        return _prompt_create()

    # Otherwise result is the project dict
    return result


# ── Caching ────────────────────────────────────────────────────────


def _get_cached_projects(*, max_age: float | None = CACHE_TTL_SECONDS) -> list[dict]:
    """Load cached projects, auto-refreshing if stale."""
    ws_name = get_active_workspace()
    if ws_name is None:
        return []
    projects, refreshed_at = load_workspace_projects(ws_name)

    if max_age is not None and projects:
        is_stale = refreshed_at is None or (time.time() - refreshed_at) > max_age
        if is_stale:
            try:
                return _refresh_projects()
            except Exception:
                console.print_error(
                    "Warning: Could not refresh project list, using cached data."
                )
                return projects

    return projects


def _refresh_projects() -> list[dict]:
    """Refresh project list from platform and update cache."""
    client = OsmosisClient()
    info = client.refresh_workspace_info()
    projects = info.get("projects", [])
    ws_name = get_active_workspace()
    if ws_name:
        save_workspace_projects(ws_name, projects)
        has_subscription = info.get("has_subscription")
        if has_subscription is not None:
            save_subscription_status(ws_name, bool(has_subscription))
    return projects


# ── Resolution (non-interactive, for commands like dataset) ────────


def _resolve_project(name_or_id: str | None, *, refresh: bool = False) -> dict:
    """Resolve a project by name or ID, using cache first.

    Priority: --project arg > $OSMOSIS_PROJECT > config.json default > error
    """
    if name_or_id is None:
        name_or_id = os.environ.get("OSMOSIS_PROJECT")

    if name_or_id is None:
        ws = get_active_workspace()
        if ws:
            default = get_default_project(ws)
            if default:
                name_or_id = default.get("project_name")

    if name_or_id is None:
        available = _get_cached_projects(max_age=None)
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

    projects = _get_cached_projects()
    if refresh or not projects:
        projects = _refresh_projects()

    target = name_or_id.lower()
    for p in projects:
        if p.get("project_name", "").lower() == target or p.get("id") == name_or_id:
            return p

    if not refresh:
        projects = _refresh_projects()
        for p in projects:
            if p.get("project_name", "").lower() == target or p.get("id") == name_or_id:
                return p

    raise CLIError(f"Project '{name_or_id}' not found in this workspace.")


def _require_auth() -> None:
    """Check that user is authenticated."""
    creds = get_valid_credentials()
    if creds is None:
        raise CLIError("Not logged in. Run 'osmosis login' first.")


def _require_subscription() -> None:
    """Check that the active workspace has an active subscription.

    Uses cached status first. If cached status is False or missing,
    refreshes from the platform to avoid blocking users who just subscribed.
    """
    ws_name = get_active_workspace()
    if not ws_name:
        return  # Will be caught by _require_auth

    cached = load_subscription_status(ws_name)
    if cached is True:
        return

    # Cached status is False or None — refresh to get the latest
    with contextlib.suppress(Exception):
        _refresh_projects()  # Also updates subscription cache

    # Re-check after refresh
    status = load_subscription_status(ws_name)
    if status is not True:
        raise CLIError(
            "Your workspace requires an active subscription for this action.\n"
            f"  Upgrade at: {PLATFORM_URL}/{ws_name}/settings/billing"
        )
