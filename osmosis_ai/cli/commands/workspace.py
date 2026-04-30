"""Platform workspace management commands."""

from __future__ import annotations

from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage platform workspaces (list, create, delete, switch).",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def workspace_default(ctx: typer.Context) -> None:
    """Manage platform workspaces. Launches interactive mode when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from osmosis_ai.cli.errors import CLIError
        from osmosis_ai.cli.output import OutputFormat, get_output_context
        from osmosis_ai.platform.cli.workspace import workspace as interactive_workspace

        output = get_output_context()
        if output.format is not OutputFormat.rich:
            raise CLIError(
                "Interactive workspace UI is unavailable in this mode. "
                "Use 'osmosis workspace list', 'osmosis workspace switch <name>', "
                "or 'osmosis workspace create <name>'.",
                code="INTERACTIVE_REQUIRED",
            )
        interactive_workspace()


@app.command("list")
def list_workspaces() -> Any:
    """List available platform workspaces."""
    from osmosis_ai.platform.cli.workspace import list_workspaces as _list_workspaces

    return _list_workspaces()


@app.command("create")
def create(
    name: str = typer.Argument(
        ..., help="Workspace name (lowercase, hyphens allowed)."
    ),
    timezone: str = typer.Option(
        "UTC", "--timezone", help="IANA timezone (e.g. America/New_York)."
    ),
) -> Any:
    """Create a new platform workspace."""
    from osmosis_ai.platform.cli.workspace import create_workspace

    return create_workspace(name=name, timezone=timezone)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Workspace name to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Delete a platform workspace."""
    from osmosis_ai.platform.cli.workspace import delete_workspace

    return delete_workspace(name=name, yes=yes)


@app.command("switch")
def switch(
    workspace: str = typer.Argument(..., help="Workspace name to switch to."),
) -> Any:
    """Switch to a different platform workspace."""
    from osmosis_ai.platform.cli.workspace import switch_workspace

    return switch_workspace(workspace=workspace)
