from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import get_output_context, serialize_dev_rollout_server
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.cli.output.result import ListColumn, ListResult, OperationResult
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.cli.utils import paginated_fetch, validate_list_options
from osmosis_ai.platform.cli.workspace_directory_context import (
    resolve_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_repo import (
    check_pinned_commit,
    summarize_local_git_state,
)


def up(*, ttl_hours: int | None, yes: bool = False) -> OperationResult:
    cwd = Path.cwd()
    if not (cwd / "main.py").is_file():
        raise CLIError(
            "Run from a rollout folder containing main.py.", code="INVALID_USAGE"
        )
    ctx = resolve_git_workspace_directory_context()
    state = summarize_local_git_state(ctx.workspace_directory)
    if state is None or not state.head_sha:
        raise CLIError("Not in a git repo with a commit.", code="VALIDATION")

    preflight = check_pinned_commit(
        workspace_directory=ctx.workspace_directory,
        git_identity=ctx.git_identity,
        commit_sha=state.head_sha,
    )
    if preflight.error:
        raise CLIError(preflight.error, code="VALIDATION")

    if state.is_dirty:
        require_confirmation(
            f"The remote server will run committed HEAD ({state.head_sha[:7]}), not your uncommitted changes. Continue?",
            yes=yes,
            default=False,
            warnings=[
                f"Working tree is dirty — uncommitted edits won't be on the server (runs commit {state.head_sha[:7]})."
            ],
        )

    repository_path = str(cwd.resolve().relative_to(ctx.workspace_directory))
    rollout_name = cwd.name
    client = OsmosisClient()
    result: dict[str, Any] = client.provision_dev_rollout_server(
        rollout_name=rollout_name,
        commit_sha=state.head_sha,
        repository_path=repository_path,
        entrypoint="main.py",
        ttl_hours=ttl_hours,
        credentials=ctx.credentials,
        git_identity=ctx.git_identity,
    )
    return OperationResult(
        operation="dev.server.up",
        status="success",
        resource=result,
        message=f"Rollout server provisioning at {result['url']} — it may take a few minutes to become ready; check with 'osmosis dev server list'.",
    )


def down(server_id: str) -> OperationResult:
    ctx = resolve_git_workspace_directory_context()
    client = OsmosisClient()
    result: dict[str, Any] = client.teardown_dev_rollout_server(
        server_id,
        credentials=ctx.credentials,
        git_identity=ctx.git_identity,
    )
    return OperationResult(
        operation="dev.server.down",
        status="success",
        resource=result,
        message=f"Stopped {server_id}",
    )


def list_servers(*, limit: int, all_: bool) -> ListResult:
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    ctx = resolve_git_workspace_directory_context()
    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching rollout servers..."):
        servers, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_dev_rollout_servers(
                limit=lim,
                offset=off,
                credentials=ctx.credentials,
                git_identity=ctx.git_identity,
            ),
            items_attr="dev_rollout_servers",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    items = [serialize_dev_rollout_server(server) for server in servers]

    return ListResult(
        title="Dev Rollout Servers",
        items=items,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        display_items=[
            {
                **item,
                "expires_at": format_local_date(item["expires_at"])
                if item["expires_at"]
                else "No expiration",
            }
            for item in items
        ],
        columns=[
            ListColumn(key="id", label="ID", ratio=2, overflow="fold"),
            ListColumn(key="name", label="Rollout", ratio=2, overflow="fold"),
            ListColumn(key="url", label="URL", ratio=4, overflow="fold"),
            ListColumn(key="expires_at", label="Expires At", no_wrap=True, ratio=2),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
        ],
    )
