"""Rollout commands: list."""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage rollouts (list).",
    no_args_is_help=True,
)


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> Any:
    """List rollouts for the current Git-scoped project."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_rollout,
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
    with output.status("Fetching rollouts..."):
        if fetch_all:
            rollouts, total_count = fetch_all_pages(
                lambda lim, off: client.list_rollouts(
                    limit=lim,
                    offset=off,
                    credentials=credentials,
                    git_identity=git_identity,
                ),
                items_attr="rollouts",
            )
            has_more = False
            next_offset = None
        else:
            page = client.list_rollouts(
                limit=effective_limit,
                offset=0,
                credentials=credentials,
                git_identity=git_identity,
            )
            rollouts = page.rollouts
            total_count = page.total_count
            has_more = page.has_more
            next_offset = page.next_offset

    return ListResult(
        title="Rollouts",
        items=[serialize_rollout(rollout) for rollout in rollouts],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="name", label="Name"),
            ListColumn(key="is_active", label="Active"),
            ListColumn(key="repo_full_name", label="Repository"),
            ListColumn(key="last_synced_commit_sha", label="Commit"),
            ListColumn(key="created_at", label="Created"),
            ListColumn(key="id", label="ID", no_wrap=True),
        ],
    )
