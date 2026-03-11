"""Handler for `osmosis model` commands."""

from __future__ import annotations

import typer

from osmosis_ai.cli.console import console

from .project import _require_auth, _resolve_project_id

app = typer.Typer(help="Manage models.")


@app.command("list")
def list_models(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    limit: int = typer.Option(50, "--limit", help="Maximum number of models to show."),
) -> None:
    """List models in a project."""
    ws_name, credentials = _require_auth()
    from osmosis_ai.platform.api.client import OsmosisClient

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()
    result = client.list_models(project_id, limit=limit, credentials=credentials)

    if not result.models:
        console.print("No models found.")
        return

    console.print(f"Models ({result.total_count}):", style="bold")
    for m in result.models:
        status_str = console.format_styled(f"[{m.status}]", "dim")
        base = m.base_model or ""
        date = m.created_at[:10] if m.created_at else ""

        console.print(f"  {m.id[:8]}  {m.model_name}  {status_str}  {base}  {date}")

    if result.has_more:
        remaining = result.total_count - len(result.models)
        console.print(f"  ... and {remaining} more")
