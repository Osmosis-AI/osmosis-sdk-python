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


def _require_confirmation(message: str, *, yes: bool) -> None:
    if yes:
        return

    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is not OutputFormat.rich or not output.interactive:
        err = CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )
        if output.format is OutputFormat.json:
            from osmosis_ai.cli.output import emit_structured_error_to_stderr

            emit_structured_error_to_stderr(err)
            raise typer.Exit(1)
        raise err

    from osmosis_ai.cli.prompts import require_confirmation

    require_confirmation(message, yes=yes)


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
) -> Any:
    """List base models in the current workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_model,
    )
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        fetch_all_pages,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    _ws_name, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching models..."):
        if fetch_all:
            models, total = fetch_all_pages(
                lambda lim, off: client.list_base_models(
                    limit=lim, offset=off, credentials=credentials
                ),
                items_attr="models",
            )
            has_more = False
            next_offset = None
        else:
            result = client.list_base_models(
                limit=effective_limit, offset=0, credentials=credentials
            )
            models = result.models
            total = result.total_count
            has_more = result.has_more
            next_offset = result.next_offset

    return ListResult(
        title="Base Models",
        items=[serialize_model(model) for model in models],
        total_count=total,
        has_more=has_more,
        next_offset=next_offset,
        columns=[
            ListColumn(key="model_name", label="Model"),
            ListColumn(key="base_model", label="Base"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="creator_name", label="Creator"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Model path (e.g. google/gemma-2-9b-it)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Delete a base model."""
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.utils import _require_auth

    _ws_name, credentials = _require_auth()
    client = OsmosisClient()
    output = get_output_context()

    try:
        with output.status("Checking model dependencies..."):
            affected = client.get_model_affected_resources(
                name, credentials=credentials
            )
    except PlatformAPIError as e:
        raise CLIError(f"Unable to verify model dependencies: {e}") from e

    if affected.has_blocking_runs:
        blocking_runs = [
            {
                "id": run.id,
                "training_run_name": run.training_run_name,
            }
            for run in affected.training_runs_using_model
        ]
        if output.format is not OutputFormat.rich:
            raise CLIError(
                "Cannot delete this model because training runs depend on it.",
                details={"training_runs": blocking_runs},
            )
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

    _require_confirmation(f'Delete model "{name}"? This cannot be undone.', yes=yes)
    with output.status("Deleting model..."):
        client.delete_model(name, credentials=credentials)
    return OperationResult(
        operation="model.delete",
        status="success",
        resource={"name": name},
        message=f'Model "{name}" deleted.',
    )
