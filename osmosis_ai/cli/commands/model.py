"""Model management commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import not_implemented
from osmosis_ai.platform.cli.constants import DEFAULT_LIST_LIMIT

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
    from osmosis_ai.platform.cli.utils import entity_status_style, format_dim_date

    if not result.models:
        return
    models = result.models if max_display is None else result.models[:max_display]
    if not models:
        return
    console.print(f"{title} ({result.total_count}):", style="bold")
    for m in models:
        short_id = console.format_styled(m.id[:8], "dim")
        style = entity_status_style(m.status) or "dim"
        status_str = console.format_styled(f"[{m.status}]", style)
        name = console.escape(m.model_name)
        meta = metadata_fn(m)
        date = format_dim_date(m.created_at)
        console.print(
            f"  {short_id}  {name}  {status_str}  {meta}  {date}",
            highlight=False,
        )
    console.print()


@app.command("list")
def list_models(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    limit: int = typer.Option(
        DEFAULT_LIST_LIMIT,
        "--limit",
        help="Maximum number of models to show per category.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all models."),
) -> None:
    """List models in a project."""
    from osmosis_ai.platform.cli.project import _require_auth, _resolve_project_id
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.models import (
        PaginatedBaseModels,
        PaginatedOutputModels,
    )

    with console.spinner("Fetching models..."):
        project_id = _resolve_project_id(project, workspace_name=ws_name)
        client = OsmosisClient()
        if fetch_all:
            base_models, base_total = fetch_all_pages(
                lambda lim, off: client.list_base_models(
                    project_id, limit=lim, offset=off, credentials=credentials
                ),
                items_attr="models",
            )
            output_models, output_total = fetch_all_pages(
                lambda lim, off: client.list_output_models(
                    project_id, limit=lim, offset=off, credentials=credentials
                ),
                items_attr="models",
            )
            base_result = PaginatedBaseModels(
                models=base_models, total_count=base_total, has_more=False
            )
            output_result = PaginatedOutputModels(
                models=output_models, total_count=output_total, has_more=False
            )
        else:
            base_result, output_result = client.fetch_all_models(
                project_id, limit=effective_limit, credentials=credentials
            )

    if not base_result.models and not output_result.models:
        console.print("No models found.")
        return

    display_limit = None if fetch_all else effective_limit

    _print_model_section(
        output_result,
        "Output Models",
        lambda m: (
            console.format_styled(f"from {m.training_run_name}", "dim")
            if m.training_run_name
            else ""
        ),
        max_display=display_limit,
    )

    _print_model_section(
        base_result,
        "Base Models",
        lambda m: (
            console.format_styled(f"by {m.creator_name}", "dim")
            if m.creator_name
            else ""
        ),
        max_display=display_limit,
    )

    if not fetch_all:
        total_shown = min(len(output_result.models), effective_limit) + min(
            len(base_result.models), effective_limit
        )
        print_pagination_footer(
            total_shown, output_result.total_count + base_result.total_count, "models"
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
def delete(
    id: str = typer.Argument(..., help="Model ID to delete."),
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a model."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.cli.project import _require_auth, _resolve_project_id

    ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    project_id = _resolve_project_id(project, workspace_name=ws_name)
    client = OsmosisClient()

    from osmosis_ai.platform.cli.utils import fetch_all_pages, resolve_id_prefix

    base_models, _ = fetch_all_pages(
        lambda lim, off: client.list_base_models(
            project_id, limit=lim, offset=off, credentials=credentials
        ),
        items_attr="models",
    )
    output_models, _ = fetch_all_pages(
        lambda lim, off: client.list_output_models(
            project_id, limit=lim, offset=off, credentials=credentials
        ),
        items_attr="models",
    )

    all_models = base_models + output_models
    model_id = resolve_id_prefix(id, all_models, entity_name="model")
    model_type = "base" if any(m.id == model_id for m in base_models) else "output"

    try:
        affected = client.get_model_affected_resources(
            model_id, project_id, model_type, credentials=credentials
        )
    except Exception as e:
        raise CLIError(f"Unable to verify model dependencies: {e}") from e

    if affected.has_blocking_runs:
        console.print(
            "Cannot delete this model — the following training runs depend on it:",
            style="red",
        )
        for run in affected.training_runs_using_model:
            name = console.escape(run.name) if run.name else "(unnamed)"
            console.print(f"  {run.id[:8]}  {name}  [{run.project_name}]")
        console.print("\nDelete these training runs first, then retry.", style="dim")
        raise typer.Exit(1)

    msg = f"Delete model {model_id[:8]}...? This cannot be undone."
    if affected.creator_training_run:
        r = affected.creator_training_run
        name = r.name or "(unnamed)"
        msg += f"\n  Note: this model was created by training run '{name}' ({r.id[:8]})"

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(msg, yes=yes)

    client.delete_model(model_id, project_id, credentials=credentials)
    console.print(f"Model {model_id[:8]} deleted.", style="green")
