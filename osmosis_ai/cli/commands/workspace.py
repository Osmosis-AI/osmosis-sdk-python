"""Workspace management commands."""

from __future__ import annotations

import typer

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
def create(
    name: str = typer.Argument(
        ..., help="Workspace name (lowercase, hyphens allowed)."
    ),
    timezone: str = typer.Option(
        "UTC", "--timezone", help="IANA timezone (e.g. America/New_York)."
    ),
) -> None:
    """Create a new workspace."""
    from osmosis_ai.cli.console import console
    from osmosis_ai.platform.cli.project import _require_auth

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    result = client.create_workspace(name, timezone, credentials=credentials)
    console.print(f"Workspace '{result['name']}' created.", style="green")


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Workspace name to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a workspace."""
    from osmosis_ai.cli.console import console
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.cli.project import _require_auth

    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    ws_data = client.list_workspaces(credentials=credentials)
    workspace = None
    for ws in ws_data.get("workspaces", []):
        if ws.get("name") == name.lower():
            workspace = ws
            break

    if not workspace:
        raise CLIError(f"Workspace '{name}' not found.")

    try:
        status = client.get_workspace_deletion_status(
            workspace["id"], credentials=credentials
        )
    except Exception as e:
        raise CLIError(f"Unable to verify workspace deletion safety: {e}") from e

    if not status.is_owner:
        raise CLIError("Only workspace owners can delete a workspace.")

    if status.is_last_workspace:
        raise CLIError("Cannot delete your only workspace.")

    if status.projects_with_running_processes:
        console.print(
            "Cannot delete workspace — the following projects have running processes:",
            style="red",
        )
        for p in status.projects_with_running_processes:
            processes = []
            if not p.feature_pipelines.valid:
                processes.append(f"{p.feature_pipelines.count} pipeline(s)")
            if not p.training_runs.valid:
                processes.append(f"{p.training_runs.count} training run(s)")
            if not p.models.valid:
                processes.append(f"{p.models.count} active model(s)")
            console.print(f"  {console.escape(p.project_name)}: {', '.join(processes)}")
        console.print("\nStop all running processes first, then retry.", style="dim")
        raise typer.Exit(1)

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(
        f"Delete workspace '{workspace['name']}'? "
        "This will delete all projects, datasets, and training runs. "
        "This cannot be undone.",
        yes=yes,
    )

    client.delete_workspace(workspace["id"], credentials=credentials)
    console.print(f"Workspace '{workspace['name']}' deleted.", style="green")


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
