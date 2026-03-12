from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.prompts import Choice, Separator, confirm, select
from osmosis_ai.platform.auth import (
    delete_workspace_credentials,
    get_all_workspaces,
    load_workspace_credentials,
)
from osmosis_ai.platform.auth.local_config import clear_workspace_data
from osmosis_ai.platform.auth.platform_client import revoke_cli_token

from .constants import LOGOUT_ALL


def _revoke_and_delete(name: str) -> bool:
    """Revoke token server-side (best-effort), then delete local credentials."""
    creds = load_workspace_credentials(name)
    if creds and not creds.is_expired():
        revoke_cli_token(creds)
    # Always attempt both cleanup steps: a 401 during revocation may have
    # already deleted the credentials via _handle_401_and_cleanup, causing
    # delete_workspace_credentials to return False.  We must still clear
    # cached workspace data regardless.
    deleted = delete_workspace_credentials(name)
    clear_workspace_data(name)
    # Report success if credentials were present — even if delete returned False,
    # they may have been removed during revocation by _handle_401_and_cleanup.
    return deleted or creds is not None


def _logout_all(
    workspaces: list[tuple[str, Any, bool]],
    skip_confirm: bool,
) -> None:
    """Logout from all workspaces."""
    workspace_names = [name for name, _, _ in workspaces]

    if not skip_confirm:
        console.print(f"This will logout from {len(workspaces)} workspace(s):")
        for name in workspace_names:
            console.print(f"  - {name}")

        result = confirm(
            f"Logout from all {len(workspaces)} workspace(s)?", default=False
        )
        if result is None:  # User cancelled with Ctrl+C
            return
        if not result:
            console.print("Cancelled.")
            return

    success_count = 0
    for name, _, _ in workspaces:
        if _revoke_and_delete(name):
            success_count += 1

    console.print(
        f"Logged out from {success_count}/{len(workspaces)} workspace(s).",
        style="green",
    )


def _logout_interactive(
    workspaces: list[tuple[str, Any, bool]],
    skip_confirm: bool,
) -> None:
    """Interactive workspace selection for logout."""
    if len(workspaces) == 1:
        # Only one workspace, logout directly
        name, _, _ = workspaces[0]
        if not skip_confirm:
            result = confirm(f"Logout from '{name}'?", default=False)
            if result is None:  # User cancelled with Ctrl+C
                return
            if not result:
                console.print("Cancelled.")
                return
        if _revoke_and_delete(name):
            console.print(f"Logged out from '{name}'.", style="green")
        return

    # Multiple workspaces, show selection
    choices = []
    for name, creds, is_active in workspaces:
        active_marker = " (active)" if is_active else ""
        expired_marker = " [expired]" if creds.is_expired() else ""
        title = f"{name}{active_marker}{expired_marker}"
        choices.append(Choice(title=title, value=name))

    choices.append(Separator())
    choices.append(Choice(title="All workspaces", value=LOGOUT_ALL))

    selected = select("Select workspace to logout from:", choices=choices)
    if selected is None:  # User cancelled with Ctrl+C
        return

    if selected == LOGOUT_ALL:
        _logout_all(workspaces, skip_confirm)
    else:
        # Selected a specific workspace
        name = selected
        if not skip_confirm:
            result = confirm(f"Logout from '{name}'?", default=False)
            if result is None:  # User cancelled with Ctrl+C
                return
            if not result:
                console.print("Cancelled.")
                return
        if _revoke_and_delete(name):
            console.print(f"Logged out from '{name}'.", style="green")


def logout(
    logout_all: bool = typer.Option(False, "--all", help="Logout from all workspaces."),
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> None:
    """Logout from Osmosis AI CLI."""
    workspaces = get_all_workspaces()

    if not workspaces:
        console.print("Not logged in.")
        return

    if logout_all:
        _logout_all(workspaces, skip_confirm)
    else:
        _logout_interactive(workspaces, skip_confirm)
