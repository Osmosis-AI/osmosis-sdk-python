"""Model management commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import not_implemented
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage models (list, deploy, export, build, delete).", no_args_is_help=True
)


def _print_model_section(
    models: list[Any],
    total_count: int,
    title: str,
    metadata_fn: Callable[[Any], str],
) -> None:
    """Print a section of models (base or output) with consistent formatting."""
    from osmosis_ai.platform.cli.utils import entity_status_style, format_dim_date

    if not models:
        return
    console.print(f"{title} ({total_count}):", style="bold")
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


def _fetch_all_models(client: Any, credentials: Any) -> tuple[list[Any], list[Any]]:
    """Fetch all base and output models via exhaustive pagination."""
    from osmosis_ai.platform.cli.utils import fetch_all_pages

    base_models, _ = fetch_all_pages(
        lambda lim, off: client.list_base_models(lim, off, credentials=credentials),
        items_attr="models",
    )
    output_models, _ = fetch_all_pages(
        lambda lim, off: client.list_output_models(lim, off, credentials=credentials),
        items_attr="models",
    )
    return base_models, output_models


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of models to show per category.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all models."),
) -> None:
    """List models in the current workspace."""
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    with console.spinner("Fetching models..."):
        client = OsmosisClient()
        if fetch_all:
            base_models, output_models = _fetch_all_models(client, credentials)
            base_total = len(base_models)
            output_total = len(output_models)
        else:
            base_result, output_result = client.fetch_all_models(
                limit=effective_limit, credentials=credentials
            )
            base_models = base_result.models
            base_total = base_result.total_count
            output_models = output_result.models
            output_total = output_result.total_count

    if not base_models and not output_models:
        console.print("No models found.")
        return

    _print_model_section(
        output_models,
        output_total,
        "Output Models",
        lambda m: (
            console.format_styled(f"from {m.training_run_name}", "dim")
            if m.training_run_name
            else ""
        ),
    )

    _print_model_section(
        base_models,
        base_total,
        "Base Models",
        lambda m: (
            console.format_styled(f"by {m.creator_name}", "dim")
            if m.creator_name
            else ""
        ),
    )

    if not fetch_all:
        total_shown = len(output_models) + len(base_models)
        print_pagination_footer(total_shown, output_total + base_total, "models")


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
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a model."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.cli.utils import _require_auth, resolve_id_prefix

    _ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    base_models, output_models = _fetch_all_models(client, credentials)
    model_id = resolve_id_prefix(id, base_models + output_models, entity_name="model")
    model_type = "base" if any(m.id == model_id for m in base_models) else "output"

    try:
        affected = client.get_model_affected_resources(
            model_id, model_type, credentials=credentials
        )
    except Exception as e:
        raise CLIError(f"Unable to verify model dependencies: {e}") from e

    if affected.has_blocking_runs:
        console.print(
            "Cannot delete this model — the following training runs depend on it:",
            style="red",
        )
        for run in affected.training_runs_using_model:
            label = (
                console.escape(run.training_run_name)
                if run.training_run_name
                else "(unnamed)"
            )
            console.print(f"  {run.id[:8]}  {label}")
        console.print("\nDelete these training runs first, then retry.", style="dim")
        raise typer.Exit(1)

    msg = f"Delete model {model_id[:8]}...? This cannot be undone."
    if affected.creator_training_run:
        r = affected.creator_training_run
        tr_name = (
            console.escape(r.training_run_name) if r.training_run_name else "(unnamed)"
        )
        msg += (
            f"\n  Note: this model was created by training run '{tr_name}' ({r.id[:8]})"
        )

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(msg, yes=yes)

    client.delete_model(model_id, credentials=credentials)
    console.print(f"Model {model_id[:8]} deleted.", style="green")
