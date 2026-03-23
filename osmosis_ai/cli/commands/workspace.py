"""Workspace management commands."""

from __future__ import annotations

import typer

from osmosis_ai.cli.errors import not_implemented

app: typer.Typer = typer.Typer(
    help="Manage workspaces (list, create, delete, switch).",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def workspace_default(ctx: typer.Context) -> None:
    """Manage workspaces. Launches interactive mode when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from osmosis_ai.platform.cli.workspace import workspace as interactive_workspace

        interactive_workspace()


@app.command("list")
def list_workspaces() -> None:
    """List available workspaces."""
    from osmosis_ai.platform.cli.workspace import list_workspaces as _list_workspaces

    _list_workspaces()


@app.command("create")
def create() -> None:
    """Create a new workspace."""
    not_implemented("workspace", "create")


@app.command("delete")
def delete() -> None:
    """Delete a workspace."""
    not_implemented("workspace", "delete")


@app.command("switch")
def switch(
    workspace: str = typer.Argument(..., help="Workspace name to switch to."),
    project: str | None = typer.Option(
        None, "--project", help="Also set the default project."
    ),
) -> None:
    """Switch to a different workspace."""
    from osmosis_ai.platform.cli.workspace import switch_workspace

    switch_workspace(workspace=workspace, project=project)
