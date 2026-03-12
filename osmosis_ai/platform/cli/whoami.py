from __future__ import annotations

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.auth import get_all_workspaces
from osmosis_ai.platform.auth.config import PLATFORM_URL


def whoami() -> None:
    """Show current authenticated user and workspace."""
    workspaces = get_all_workspaces()

    if not workspaces:
        console.print(
            "Not logged in. Run 'osmosis login' to authenticate.", style="yellow"
        )
        raise typer.Exit(1)

    # Find active workspace for user info
    active_creds = None
    for _, creds, is_active in workspaces:
        if is_active:
            active_creds = creds
            break

    # Show user info from active workspace
    if active_creds:
        rows = [("Email", active_creds.user.email)]
        if active_creds.user.name:
            rows.append(("Name", active_creds.user.name))
        console.table(rows)

    # Show all workspaces
    console.print(f"\nWorkspaces ({len(workspaces)}):", style="bold")
    for name, creds, is_active in workspaces:
        url = f"{PLATFORM_URL}/{name}"
        if creds.is_expired():
            # Expired workspace: dim/red + name + role + [expired]
            # Show expired indicator regardless of active status
            line = console.format_styled("●", "red dim")
            line += console.format_styled(
                f" {name} ({creds.organization.role}) [expired]", "dim"
            )
            console.print(line)
        elif is_active:
            # Active workspace: green bullet + name + role
            line = console.format_styled("●", "green")
            line += f" {name} ({creds.organization.role})"
            line += f"  {console.format_styled(url, 'dim')}"
            console.print(line)
        else:
            # Inactive but valid workspace
            line = console.format_styled("○", "dim")
            line += f" {name} ({creds.organization.role})"
            line += f"  {console.format_styled(url, 'dim')}"
            console.print(line)
