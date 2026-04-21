"""Rollout commands: serve, list."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from osmosis_ai.cli.console import Console

app: typer.Typer = typer.Typer(
    help="Manage rollouts (serve, list).",
    no_args_is_help=True,
)

# Valid log levels for uvicorn (defined locally to avoid importing the heavy
# serve module at CLI parse time).
LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


def _json_error_response(*, status_code: int, detail: str):
    from starlette.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


def _format_http_origin(host: str, port: int) -> str:
    """Format ``http://host:port`` for display and probes; bracket IPv6 literals."""
    import ipaddress

    h = host.strip()
    if ":" in h:
        try:
            ipaddress.IPv6Address(h)
        except ValueError:
            pass
        else:
            return f"http://[{h}]:{port}"
    return f"http://{h}:{port}"


def _trace_log_basename(rollout_id: str) -> str:
    """Flat, path-safe trace filename; original id is still logged in JSONL events."""
    digest = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()[:16]
    stem = rollout_id.replace("\\", "/").replace("/", "_")
    stem = re.sub(r"[^0-9A-Za-z._-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    while ".." in stem:
        stem = stem.replace("..", "_")
    if not stem:
        stem = "rollout"
    stem = stem[:48]
    return f"{stem}_{digest}.jsonl"


@lru_cache(maxsize=1)
def _tracing_backend_proxy_cls() -> type:
    """Lazily define tracing proxy so importing this module stays lightweight."""
    from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
    from osmosis_ai.rollout.types import ExecutionRequest, ExecutionResult

    class _TracingBackendProxy(ExecutionBackend):
        """Thin proxy that JSONL-logs rollout execution events (CLI-layer tracing)."""

        def __init__(self, inner: ExecutionBackend, session_dir: Path) -> None:
            self._inner = inner
            self.session_dir = session_dir
            self.session_dir.mkdir(parents=True, exist_ok=True)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def max_concurrency(self) -> int:
            return self._inner.max_concurrency

        def health(self) -> dict[str, Any]:
            return self._inner.health()

        async def execute(
            self,
            request: ExecutionRequest,
            on_workflow_complete: ResultCallback,
            on_grader_complete: ResultCallback | None = None,
        ) -> None:
            import json

            log_path = self.session_dir / _trace_log_basename(request.id)

            def _append(event: dict[str, Any]) -> None:
                event.setdefault("ts", time.time())
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")

            _append(
                {
                    "event": "rollout_start",
                    "rollout_id": request.id,
                    "label": request.label,
                }
            )

            async def _wf_wrap(result: ExecutionResult) -> None:
                _append(
                    {
                        "event": "workflow_complete",
                        "status": str(result.status),
                        "err_message": result.err_message,
                    }
                )
                await on_workflow_complete(result)

            async def _gr_wrap(result: ExecutionResult) -> None:
                _append(
                    {
                        "event": "grader_complete",
                        "status": str(result.status),
                        "err_message": result.err_message,
                    }
                )
                if on_grader_complete:
                    await on_grader_complete(result)

            await self._inner.execute(
                request,
                _wf_wrap,
                _gr_wrap if on_grader_complete else None,
            )

    return _TracingBackendProxy


def _serve(
    *,
    config_path: Path,
    port: int | None,
    host: str | None,
    no_validate: bool,
    validate_only: bool,
    log_level: LogLevel | None,
    console: Console,
) -> None:
    if validate_only and no_validate:
        console.print_error(
            "Error: --validate-only and --no-validate are mutually exclusive."
        )
        raise typer.Exit(1)

    from osmosis_ai.platform.cli.serve_config import load_serve_config

    try:
        config = load_serve_config(Path(config_path))
    except CLIError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None

    updates: dict[str, object] = {}
    if port is not None:
        updates["server_port"] = port
    if host is not None:
        updates["server_host"] = host
    if log_level is not None:
        updates["server_log_level"] = log_level
    if updates:
        config = config.model_copy(update=updates)

    serve_host = config.server_host
    serve_port = config.server_port
    uvicorn_log_level = config.server_log_level

    from osmosis_ai.eval.common.cli import auto_discover_grader, load_workflow

    workflow_cls, workflow_config, entrypoint_module, wf_err = load_workflow(
        rollout=config.serve_rollout,
        entrypoint=config.serve_entrypoint,
        quiet=False,
        console=console,
    )
    if wf_err or workflow_cls is None:
        console.print_error(wf_err or "Failed to load workflow.")
        raise typer.Exit(1)
    assert entrypoint_module is not None

    try:
        grader_cls, grader_config = auto_discover_grader(
            entrypoint_module,
            entrypoint_label=config.serve_entrypoint,
        )
    except CLIError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None

    if grader_cls is None:
        console.print_error(
            "No Grader was found in the entrypoint module. "
            "`osmosis rollout serve` requires a concrete Grader (and typically a "
            "GraderConfig) alongside the workflow; this cannot be skipped."
        )
        raise typer.Exit(1)

    do_validate = not (no_validate or config.debug_no_validate)
    if validate_only:
        do_validate = True

    from osmosis_ai.rollout.validator import validate_backend

    if do_validate:
        v_result = validate_backend(
            workflow_cls,
            workflow_config,
            grader_cls=grader_cls,
            grader_config=grader_config,
        )
        for w in v_result.warnings:
            console.print(f"[WARN] [{w.code}] {w.message}")
        for err in v_result.errors:
            console.print_error(f"[{err.code}] {err.message}")
        if not v_result.valid:
            raise typer.Exit(1)

    if validate_only:
        console.print("Validation passed.")
        return

    from osmosis_ai.rollout.validator import resolved_agent_name

    agent_name = resolved_agent_name(workflow_cls, workflow_config)

    from osmosis_ai.rollout.backend.local import LocalBackend

    backend = LocalBackend(
        workflow=workflow_cls,
        workflow_config=workflow_config,
        grader=grader_cls,
        grader_config=grader_config,
    )

    trace_session_path: Path | None = None
    if config.debug_trace_dir:
        trace_root = Path(config.debug_trace_dir)
        trace_session_path = trace_root / (
            f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"
        )
        backend = _tracing_backend_proxy_cls()(backend, trace_session_path)

    from osmosis_ai.rollout.server.app import create_rollout_server

    fastapi_app = create_rollout_server(backend=backend)

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class _RolloutLabelMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.method != "POST" or request.url.path != "/rollout":
                return await call_next(request)
            body = await request.body()
            import json

            async def receive_replay():
                return {"type": "http.request", "body": body, "more_body": False}

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return await call_next(
                    Request(request.scope, receive_replay),
                )

            if not isinstance(data, dict) or data.get("label") in (None, ""):
                return _json_error_response(
                    status_code=400,
                    detail=(
                        "Missing required field 'label' for rollout requests when a "
                        "grader is configured."
                    ),
                )

            return await call_next(Request(request.scope, receive_replay))

    fastapi_app.add_middleware(_RolloutLabelMiddleware)

    grader_name = grader_cls.__name__ if grader_cls else "(none)"

    trace_display = (
        str(trace_session_path) if trace_session_path is not None else "(off)"
    )

    console.table(
        [
            ("Agent", agent_name),
            ("Address", _format_http_origin(serve_host, serve_port)),
            ("Grader", grader_name),
            ("Trace", trace_display),
        ],
        title="Rollout Server",
    )

    import uvicorn

    uvicorn.run(
        fastapi_app,
        host=serve_host,
        port=serve_port,
        log_level=str(uvicorn_log_level),
    )


@app.command("serve")
def serve(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to serve TOML config file.",
    ),
    port: int | None = typer.Option(
        None,
        "-p",
        "--port",
        help="Port to bind to (overrides config).",
    ),
    host: str | None = typer.Option(
        None,
        "-H",
        "--host",
        help="Host to bind to (overrides config).",
    ),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip backend validation."
    ),
    validate_only: bool = typer.Option(
        False, "--validate-only", help="Validate workflow/grader and exit."
    ),
    log_level: LogLevel | None = typer.Option(
        None,
        "--log-level",
        help="Uvicorn log level (overrides config).",
    ),
) -> None:
    """Start a RolloutServer from a TOML config file."""
    from osmosis_ai.cli.console import Console

    _serve(
        config_path=config_path,
        port=port,
        host=host,
        no_validate=no_validate,
        validate_only=validate_only,
        log_level=log_level,
        console=Console(),
    )


def _format_commit(r: Any) -> str:
    """Format commit SHA as a clickable terminal hyperlink (OSC 8 via Rich)."""
    sha = r.last_synced_commit_sha
    if not sha:
        return "[dim]—[/dim]"
    short = sha[:7]
    if r.repo_full_name:
        url = f"https://github.com/{r.repo_full_name}/tree/{sha}"
        return f"[dim][link={url}]{short}[/link][/dim]"
    return f"[dim]{short}[/dim]"


@app.command("list")
def list_rollouts(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of rollouts to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollouts."),
) -> None:
    """List rollouts in the current workspace."""
    from osmosis_ai.cli.console import Console
    from osmosis_ai.platform.cli.utils import (
        _require_auth,
        format_dim_date,
        paginated_fetch,
        print_pagination_footer,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    console = Console()
    _, credentials = _require_auth()

    from osmosis_ai.platform.api.client import OsmosisClient

    with console.spinner("Fetching rollouts..."):
        client = OsmosisClient()
        rollouts, total_count, _has_more = paginated_fetch(
            lambda lim, off: client.list_rollouts(
                limit=lim, offset=off, credentials=credentials
            ),
            items_attr="rollouts",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    if not rollouts:
        console.print("No rollouts found.")
        return

    console.print(f"Rollouts ({total_count}):", style="bold")
    for r in rollouts:
        name = console.escape(r.name)
        active = (
            console.format_styled("[active]", "green")
            if r.is_active
            else console.format_styled("[inactive]", "dim")
        )
        commit = _format_commit(r)
        date = format_dim_date(r.created_at)
        console.print(
            f"  {name}  {active}  {commit}  {date}",
            highlight=False,
        )

    print_pagination_footer(len(rollouts), total_count, "rollouts")
