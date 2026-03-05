from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.prompts import Choice, Separator, confirm, select
from osmosis_ai.platform.auth import (
    delete_workspace_credentials,
    get_all_workspaces,
)
from osmosis_ai.platform.auth.local_config import clear_workspace_data


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
        if delete_workspace_credentials(name):
            clear_workspace_data(name)
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
        if delete_workspace_credentials(name):
            clear_workspace_data(name)
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
    choices.append(Choice(title="All workspaces", value="__all__"))

    selected = select("Select workspace to logout from:", choices=choices)
    if selected is None:  # User cancelled with Ctrl+C
        return

    if selected == "__all__":
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
        if delete_workspace_credentials(name):
            clear_workspace_data(name)
            console.print(f"Logged out from '{name}'.", style="green")


def logout(
    logout_all: bool = typer.Option(False, "--all", help="Logout from all workspaces."),
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> None:
    """Logout and revoke CLI token."""
    workspaces = get_all_workspaces()

    if not workspaces:
        console.print("Not logged in.")
        return

    if logout_all:
        _logout_all(workspaces, skip_confirm)
    else:
        _logout_interactive(workspaces, skip_confirm)
