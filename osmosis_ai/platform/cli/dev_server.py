from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import get_output_context, serialize_dev_rollout_server
from osmosis_ai.cli.output.context import OutputFormat
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.cli.output.result import ListColumn, ListResult, OperationResult
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import LogEntry
from osmosis_ai.platform.cli.utils import paginated_fetch, validate_list_options
from osmosis_ai.platform.cli.workspace_directory_context import (
    resolve_git_workspace_directory_context,
)
from osmosis_ai.platform.cli.workspace_repo import (
    check_pinned_commit,
    summarize_local_git_state,
)
from osmosis_ai.platform.constants import DevServerBackend, DevServerSandboxEnvironment


def up(
    *,
    ttl_hours: int | None,
    yes: bool = False,
    sandbox_environment: DevServerSandboxEnvironment | None = None,
    backend: DevServerBackend | None = None,
    url: str | None = None,
    path: str = "tasks",
    ref: str | None = None,
) -> OperationResult:
    if url is not None:
        return up_source(
            url=url,
            path=path,
            ref=ref,
            ttl_hours=ttl_hours,
            backend=backend,
            sandbox_environment=sandbox_environment,
        )
    cwd = Path.cwd()
    if not (cwd / "main.py").is_file():
        raise CLIError(
            "Run from a rollout folder containing main.py.", code="VALIDATION"
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
    options: dict[str, Any] = {"backend": backend} if backend is not None else {}
    result: dict[str, Any] = client.provision_dev_rollout_server(
        rollout_name=rollout_name,
        commit_sha=state.head_sha,
        repository_path=repository_path,
        entrypoint="main.py",
        ttl_hours=ttl_hours,
        credentials=ctx.credentials,
        git_identity=ctx.git_identity,
        sandbox_environment=sandbox_environment,
        **options,
    )
    unconfirmed = None
    if backend is not None and result.get("backend") != backend.value:
        unconfirmed = "deployment backend"
    elif (
        sandbox_environment is not None
        and result.get("sandbox_environment") != sandbox_environment.value
    ):
        unconfirmed = "sandbox environment"
    if unconfirmed:
        server_id = result["id"]
        try:
            client.teardown_dev_rollout_server(
                server_id,
                credentials=ctx.credentials,
                git_identity=ctx.git_identity,
            )
        except Exception:
            raise CLIError(
                f"The platform did not confirm the requested {unconfirmed} "
                "and teardown could not be requested. Stop the server with "
                f"'osmosis dev server down {server_id}' before retrying on an "
                "updated platform.",
                code="VALIDATION",
            ) from None
        raise CLIError(
            f"The platform did not confirm the requested {unconfirmed}. "
            f"Teardown requested for server {server_id}. Update the platform "
            "before retrying and check 'osmosis dev server list' for cleanup.",
            code="VALIDATION",
        )
    api_key = result.get("api_key")
    return OperationResult(
        operation="dev.server.up",
        status="success",
        resource=result,
        message=f"Rollout server provisioning at {result['url']} — it may take a few minutes to become ready; check with 'osmosis dev server list'.",
        display_next_steps=(
            [f"api_key: {api_key}"] if isinstance(api_key, str) and api_key else []
        ),
    )


def up_source(
    *,
    url: str,
    path: str,
    ref: str | None,
    ttl_hours: int | None,
    backend: DevServerBackend | None,
    sandbox_environment: DevServerSandboxEnvironment | None,
) -> OperationResult:
    from dataclasses import asdict

    from osmosis_ai.harbor_images import TaskSource
    from osmosis_ai.platform.cli.workspace_repo import normalize_git_identity

    identity = normalize_git_identity(url).identity
    try:
        source = TaskSource(f"https://github.com/{identity}", path, ref or "")
    except ValueError as error:
        raise CLIError(str(error), code="VALIDATION") from None
    backend = backend or DevServerBackend.GKE
    sandbox_environment = sandbox_environment or DevServerSandboxEnvironment.OPENSANDBOX
    if (
        backend != DevServerBackend.GKE
        or sandbox_environment != DevServerSandboxEnvironment.OPENSANDBOX
    ):
        raise CLIError(
            "Source gateways require --backend gke --sandbox-environment opensandbox",
            code="VALIDATION",
        )
    client = OsmosisClient()
    result = client.provision_dev_rollout_server(
        rollout_name="harbor-source",
        commit_sha=source.revision,
        repository_path=".",
        entrypoint="/opt/osmosis/harbor/source_main.py",
        ttl_hours=ttl_hours,
        git_identity=identity,
        backend=backend,
        sandbox_environment=sandbox_environment,
        task_source=asdict(source),
    )
    if (
        result.get("task_source") != asdict(source)
        or result.get("backend") != backend.value
        or result.get("sandbox_environment") != sandbox_environment.value
    ):
        try:
            client.teardown_dev_rollout_server(result["id"], git_identity=identity)
        except Exception:
            raise CLIError(
                f"Platform did not confirm source configuration and teardown failed; stop server {result['id']} before retrying",
                code="VALIDATION",
            ) from None
        raise CLIError(
            "Platform did not confirm source gateway configuration; teardown requested",
            code="VALIDATION",
        )
    return OperationResult(
        operation="dev.server.up",
        status="success",
        resource=result,
        message=f"Source gateway provisioning at {result['url']}; readiness requires all images and agent prewarm to succeed.",
        display_next_steps=[f"api_key: {result['api_key']}"]
        if result.get("api_key")
        else [],
    )


def server_scope(url: str | None) -> dict[str, Any]:
    if url:
        from osmosis_ai.platform.cli.workspace_repo import normalize_git_identity

        return {"git_identity": normalize_git_identity(url).identity}
    ctx = resolve_git_workspace_directory_context()
    return {"credentials": ctx.credentials, "git_identity": ctx.git_identity}


def down(server_id: str, *, url: str | None = None) -> OperationResult:
    scope = server_scope(url)
    client = OsmosisClient()
    result: dict[str, Any] = client.teardown_dev_rollout_server(
        server_id,
        **scope,
    )
    return OperationResult(
        operation="dev.server.down",
        status="success",
        resource=result,
        message=f"Stopping rollout server {server_id}",
    )


def list_servers(*, limit: int, all_: bool, url: str | None = None) -> ListResult:
    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    scope = server_scope(url)
    output = get_output_context()
    client = OsmosisClient()
    with output.status("Fetching rollout servers..."):
        servers, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_dev_rollout_servers(
                limit=lim,
                offset=off,
                **scope,
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


def _emit_entries(entries: list[LogEntry], fmt: OutputFormat) -> None:
    if not entries:
        return
    if fmt is OutputFormat.json:
        for entry in entries:
            sys.stdout.write(
                json.dumps({"timestamp": entry.timestamp, "message": entry.message})
                + "\n"
            )
        sys.stdout.flush()
        return
    if fmt is OutputFormat.plain:
        for entry in entries:
            sys.stdout.write(f"{entry.timestamp}\t{entry.message}\n")
        sys.stdout.flush()
        return

    for entry in entries:
        console.print(
            f"{entry.timestamp}  {entry.message}", markup=False, highlight=False
        )


def logs(
    server_id: str, *, follow: bool | None, tail: int, url: str | None = None
) -> NoReturn:
    """Show logs for a remote rollout server.

    In rich mode logs stream live (attach); with ``--json``/``--plain`` the last
    ``tail`` lines are printed and the command exits unless ``follow`` is forced.
    """
    scope = server_scope(url)
    output = get_output_context()
    fmt = output.format
    effective_follow = follow if follow is not None else (fmt is OutputFormat.rich)

    client = OsmosisClient()

    if effective_follow:
        # The stream sends the last `tail` lines first, then pushes new ones.
        try:
            for entry in client.stream_dev_rollout_server_logs(
                server_id,
                tail=tail,
                **scope,
            ):
                _emit_entries([entry], fmt)
        except KeyboardInterrupt:
            if fmt is OutputFormat.rich:
                console.print(f"\nDetached from {server_id}.", style="dim")
            raise
    else:
        page = client.get_dev_rollout_server_logs(
            server_id,
            limit=tail,
            **scope,
        )
        _emit_entries(page.logs, fmt)

    output.output_emitted = True
    raise typer.Exit(0)
