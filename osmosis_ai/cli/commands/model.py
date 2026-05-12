"""Base-model management commands.

LoRA deployments live under ``osmosis deployment`` — this group is
scoped to base (foundation) models only.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage base models (list).", no_args_is_help=True)


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of base models to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all base models."),
) -> Any:
    """List base models for the current Git-scoped project."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_model,
    )
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        git_result_context,
        require_git_project_context,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_project_context()
    credentials = context.credentials
    git_identity = context.git_identity

    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching models..."):
        if fetch_all:
            models, total = fetch_all_pages(
                lambda lim, off: client.list_base_models(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=git_identity,
                ),
                items_attr="models",
            )
            has_more = False
            next_offset = None
        else:
            result = client.list_base_models(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                git_identity=git_identity,
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
        extra=git_result_context(context),
        columns=[
            ListColumn(key="model_name", label="Model"),
            ListColumn(key="base_model", label="Base"),
            ListColumn(key="status", label="Status"),
            ListColumn(key="creator_name", label="Creator"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )
