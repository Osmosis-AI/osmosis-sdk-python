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


def _fetch_all_models(client: Any, credentials: Any) -> list[Any]:
    """Fetch all models via exhaustive pagination."""
    from osmosis_ai.platform.cli.utils import fetch_all_pages

    models, _ = fetch_all_pages(
        lambda lim, off: client.list_base_models(lim, off, credentials=credentials),
        items_attr="models",
    )
    return models


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
            models = _fetch_all_models(client, credentials)
            total = len(models)
        else:
            result = client.list_base_models(
                limit=effective_limit, credentials=credentials
            )
            models = result.models
            total = result.total_count

    if not models:
        console.print("No models found.")
        return

    _print_model_section(
        models,
        total,
        "Models",
        lambda m: (
            console.format_styled(f"by {m.creator_name}", "dim")
            if m.creator_name
            else ""
        ),
    )

    if not fetch_all:
        print_pagination_footer(len(models), total, "models")


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

    models = _fetch_all_models(client, credentials)
    model_id = resolve_id_prefix(id, models, entity_name="model")

    try:
        affected = client.get_model_affected_resources(
            model_id, credentials=credentials
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

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(msg, yes=yes)

    client.delete_model(model_id, credentials=credentials)
    console.print(f"Model {model_id[:8]} deleted.", style="green")
