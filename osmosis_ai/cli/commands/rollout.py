"""Rollout commands: list."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from osmosis_ai.cli.output import CommandResult
else:
    CommandResult = Any

app: typer.Typer = typer.Typer(
    help="Manage rollouts (init, list).",
    no_args_is_help=True,
)


@app.command("init")
def init(
    name: str = typer.Argument(
        ...,
        help="Rollout name (lowercase letters, digits, and hyphens).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help=(
            "Overwrite existing rollouts/<name>/ directory and configs/{eval,training}/"
            "<name>.toml. Without --force, the command refuses to clobber existing paths."
        ),
    ),
) -> CommandResult | None:
    """Scaffold a new rollout from the workspace template placeholders.

    Creates ``rollouts/<name>/{main.py,pyproject.toml,README.md}`` and
    ``configs/{eval,training}/<name>.toml`` so you can start editing right away.
    Must run inside an Osmosis workspace directory.
    """
    from osmosis_ai.templates.init import init_command

    return init_command(name=name, force=force)


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> Any:
    """List rollouts for the current workspace directory."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        get_output_context,
        serialize_rollout,
    )
    from osmosis_ai.cli.output.display import format_local_date
    from osmosis_ai.platform.cli.utils import (
        fetch_all_pages,
        require_git_workspace_directory_context,
        validate_list_options,
    )
    from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
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
                key="repo_full_name",
                label="Repo",
                no_wrap=True,
                overflow="ellipsis",
                ratio=2,
                min_width=10,
                max_width=10,
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
