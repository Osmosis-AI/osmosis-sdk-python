"""Base-model management commands.

LoRA deployments live under ``osmosis deployment`` — this group is
scoped to base (foundation) models only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage base models (list, delete).", no_args_is_help=True
)


def _print_model_section(
    models: list[Any],
    total_count: int,
    title: str,
    metadata_fn: Callable[[Any], str],
) -> None:
    """Print a section of base models with consistent formatting."""
    from osmosis_ai.platform.cli.utils import entity_status_style, format_dim_date

    if not models:
        return
    console.print(f"{title} ({total_count}):", style="bold")
    for m in models:
        style = entity_status_style(m.status) or "dim"
        status_str = console.format_styled(f"[{m.status}]", style)
        name = console.escape(m.base_model or m.model_name)
        meta = metadata_fn(m)
        date = format_dim_date(m.created_at)
        console.print(
            f"  {name}  {status_str}  {meta}  {date}",
            highlight=False,
        )
    console.print()


def _fetch_all_models(client: Any, credentials: Any) -> list[Any]:
    """Fetch all base models via exhaustive pagination."""
    from osmosis_ai.platform.cli.utils import fetch_all_pages

    models, _ = fetch_all_pages(
        lambda lim, off: client.list_base_models(
            limit=lim, offset=off, credentials=credentials
        ),
        items_attr="models",
    )
    return models


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of base models to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all base models."),
) -> None:
    """List base models in the current workspace."""
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
        "Base Models",
        lambda m: (
            console.format_styled(f"by {m.creator_name}", "dim")
            if m.creator_name
            else ""
        ),
    )

    if not fetch_all:
        print_pagination_footer(len(models), total, "base models")


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Model path (e.g. google/gemma-2-9b-it)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a base model."""
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.errors import CLIError
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.utils import _require_auth

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()

    try:
        affected = client.get_model_affected_resources(name, credentials=credentials)
    except PlatformAPIError as e:
        raise CLIError(f"Unable to verify model dependencies: {e}") from e

    if affected.has_blocking_runs:
        console.print(
            "Cannot delete this model — the following training runs depend on it:",
            style="red",
        )
        for run in affected.training_runs_using_model:
            short_id = run.id[:8]
            label = (
                console.escape(run.training_run_name)
                if run.training_run_name
                else f"(unnamed: {short_id})"
            )
            console.print(f"  {label}  {console.format_styled(short_id, 'dim')}")
        console.print("\nDelete these training runs first, then retry.", style="dim")
        raise typer.Exit(1)

    require_confirmation(f'Delete model "{name}"? This cannot be undone.', yes=yes)
    client.delete_model(name, credentials=credentials)
    console.print(
        f'Model "{console.escape(name)}" deleted.',
        style="green",
        highlight=False,
    )
