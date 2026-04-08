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
    from osmosis_ai.platform.cli.utils import _require_auth

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
    from osmosis_ai.platform.cli.utils import _require_auth

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

    from osmosis_ai.cli.prompts import require_confirmation

    if status.has_running_processes:
        console.print(
            "This workspace has running processes:",
            style="yellow",
        )
        processes: list[str] = []
        if not status.feature_pipelines.valid:
            processes.append(f"{status.feature_pipelines.count} pipeline(s)")
        if not status.training_runs.valid:
            processes.append(f"{status.training_runs.count} training run(s)")
        if not status.models.valid:
            processes.append(f"{status.models.count} active model(s)")
        if processes:
            console.print(f"  {', '.join(processes)}")
        console.print()

    require_confirmation(
        f"Delete workspace '{workspace['name']}'? "
        "This will stop all running processes and delete all datasets and training "
        "runs. This cannot be undone.",
        yes=yes,
    )

    client.delete_workspace(workspace["id"], credentials=credentials)
    console.print(f"Workspace '{workspace['name']}' deleted.", style="green")


@app.command("switch")
def switch(
    workspace: str = typer.Argument(..., help="Workspace name to switch to."),
) -> None:
    """Switch to a different workspace."""
    from osmosis_ai.platform.cli.workspace import switch_workspace

    switch_workspace(workspace=workspace)
