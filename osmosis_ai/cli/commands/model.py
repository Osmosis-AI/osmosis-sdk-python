"""Model management commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import not_implemented

app: typer.Typer = typer.Typer(
    help="Manage models (list, deploy, export, build, delete).", no_args_is_help=True
)


def _print_model_section(
    result: Any,
    title: str,
    metadata_fn: Callable[[Any], str],
    max_display: int | None = None,
) -> None:
    """Print a section of models (base or output) with consistent formatting."""
    from osmosis_ai.platform.cli.utils import format_date

    if not result.models:
        return
    models = result.models if max_display is None else result.models[:max_display]
    if not models:
        return
    console.print(f"{title} ({result.total_count}):", style="bold")
    for m in models:
        status_str = console.format_styled(f"[{m.status}]", "dim")
        name = console.escape(m.model_name)
        meta = metadata_fn(m)
        date = format_date(m.created_at)
        console.print(f"  {m.id[:8]}  {name}  {status_str}  {meta}  {date}")
    if len(models) < result.total_count:
        remaining = result.total_count - len(models)
        console.print(f"  ... and {remaining} more")
    console.print()


@app.command("list")
def list_models(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    limit: int = typer.Option(
        50, "--limit", help="Maximum number of models to show per category."
    ),
) -> None:
    """List models in a project."""
    from osmosis_ai.platform.cli.project import _require_auth, _resolve_project_id

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()
    base_result, output_result = client.fetch_all_models(
        project_id, limit=limit, credentials=credentials
    )

    if not base_result.models and not output_result.models:
        console.print("No models found.")
        return

    _print_model_section(
        output_result,
        "Output Models",
        lambda m: (
            console.format_styled(f"from {m.training_run_name}", "dim")
            if m.training_run_name
            else ""
        ),
        max_display=limit,
    )

    _print_model_section(
        base_result,
        "Base Models",
        lambda m: (
            console.format_styled(f"by {m.creator_name}", "dim")
            if m.creator_name
            else ""
        ),
        max_display=limit,
    )


@app.command("deploy")
def deploy() -> None:
    """Deploy a model."""
    not_implemented("model", "deploy")


@app.command("export")
def export() -> None:
    """Export a model."""
    not_implemented("model", "export")


@app.command("build")
def build() -> None:
    """Build a model."""
    not_implemented("model", "build")


@app.command("delete")
def delete() -> None:
    """Delete a model."""
    not_implemented("model", "delete")
