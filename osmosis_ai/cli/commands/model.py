"""Base-model management commands.

LoRA deployments live under ``osmosis deployment`` — this group is
scoped to base (foundation) models only.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(help="Manage base models (list).", no_args_is_help=True)


def _workspace_result_context(workspace: Any) -> dict[str, Any]:
    return {
        "workspace": {"id": workspace.workspace_id, "name": workspace.workspace_name},
        "project_root": str(workspace.project_root),
    }


@app.command("list")
def list_models(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of base models to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all base models."),
) -> Any:
    """List base models in the linked project workspace."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_model,
    )
    from osmosis_ai.cli.output.display import created_column_label, format_local_date
    from osmosis_ai.platform.cli.utils import (
        entity_status_style,
        fetch_all_pages,
        require_workspace_context,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    workspace = require_workspace_context()
    credentials = workspace.credentials
    workspace_id = workspace.workspace_id

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
                    workspace_id=workspace_id,
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
                workspace_id=workspace_id,
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
        extra=_workspace_result_context(workspace),
        columns=[
            ListColumn(key="model_name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="base_model", label="Base", ratio=2, overflow="fold"),
            ListColumn(
                key="created_at",
                label=created_column_label(),
                no_wrap=True,
                ratio=1,
            ),
        ],
        display_items=[
            {
                **serialize_model(model),
                "status": (
                    console.format_styled(model.status, status_style)
                    if (status_style := entity_status_style(model.status))
                    else console.escape(model.status)
                ),
                "created_at": format_local_date(model.created_at),
            }
            for model in models
        ],
    )
