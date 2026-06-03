"""Handlers for `osmosis rollout` subcommands.

The ``cli/commands/rollout.py`` shell delegates here for ``list``; the
``init`` scaffolding verb delegates directly to ``templates.init``.
"""

from __future__ import annotations

from osmosis_ai.cli.output import (
    ListColumn,
    ListResult,
    get_output_context,
    serialize_rollout,
)
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.cli.utils import (
    paginated_fetch,
    require_git_workspace_directory_context,
    validate_list_options,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context


def list_rollouts(*, limit: int, all_: bool) -> ListResult:
    """List rollouts for the current workspace directory."""
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching rollouts..."):
        rollouts, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_rollouts(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=git_identity,
            ),
            items_attr="rollouts",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    items = [serialize_rollout(rollout) for rollout in rollouts]

    return ListResult(
        title="Rollouts",
        items=items,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(
                key="name",
                label="Name",
                ratio=6,
                overflow="fold",
                min_width=20,
            ),
            ListColumn(
                key="is_active",
                label="Active",
                no_wrap=True,
                min_width=6,
                max_width=6,
            ),
            ListColumn(
                key="last_synced_commit_sha",
                label="Commit",
                no_wrap=True,
                min_width=8,
                max_width=8,
            ),
            ListColumn(
                key="created_at",
                label="Created",
                no_wrap=True,
                min_width=10,
                max_width=10,
            ),
        ],
        display_items=[
            {
                **item,
                "is_active": "yes" if item["is_active"] else "no",
                "last_synced_commit_sha": (item["last_synced_commit_sha"] or "")[:8],
                "created_at": format_local_date(item["created_at"]).split(" ", 1)[0],
            }
            for item in items
        ],
    )
