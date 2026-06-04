"""Handlers for `osmosis model` subcommands (base/foundation models only).

LoRA deployments live under ``osmosis deployment`` — this module is scoped to
base (foundation) models only. The ``cli/commands/model.py`` shell delegates
here, mirroring the train/eval/dataset/secret command groups.
"""

from __future__ import annotations

from osmosis_ai.cli.output import (
    ListColumn,
    ListResult,
    get_output_context,
    serialize_model,
)
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.cli.utils import (
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context


def list_models(*, limit: int, all_: bool) -> ListResult:
    """List base models for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching models..."):
        models, total, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_base_models(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=git_identity,
            ),
            items_attr="models",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Base Models",
        items=[serialize_model(model) for model in models],
        total_count=total,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="model_name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="created_at", label="Added", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Added By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_model(model),
                "created_at": format_local_date(model.created_at),
                "creator_name": model.creator_name or "—",
            }
            for model in models
        ],
    )
