from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.result import OperationResult
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.cli.workspace_directory_context import (
    resolve_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_repo import (
    git_worktree_top_level,
    summarize_local_git_state,
)


def up(*, ttl_hours: int | None) -> OperationResult:
    ctx = resolve_git_workspace_directory_context()
    cwd = Path.cwd()
    state = summarize_local_git_state(cwd)
    if state is None or not state.head_sha:
        raise CLIError("Not in a git repo with a commit.", code="VALIDATION")
    if state.is_dirty:
        raise CLIError(
            "Commit and push your changes first (working tree is dirty).",
            code="VALIDATION",
        )
    root = git_worktree_top_level(cwd) or cwd
    repository_path = str(cwd.relative_to(root))
    rollout_name = cwd.name
    if not (cwd / "main.py").is_file():
        raise CLIError(
            "Run from a rollout folder containing main.py.", code="INVALID_USAGE"
        )
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
        message=f"Rollout server: {result['url']}",
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
