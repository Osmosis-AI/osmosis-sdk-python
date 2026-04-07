"""Rollout commands: serve, test, list."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from osmosis_ai.cli.errors import CLIError, not_implemented

ProbeOutcome = Literal["ready", "exhausted", "shutdown"]

if TYPE_CHECKING:
    from osmosis_ai.cli.console import Console

app: typer.Typer = typer.Typer(
    help="Manage rollouts (serve, test, list).",
    no_args_is_help=True,
)

# Valid log levels for uvicorn (defined locally to avoid importing the heavy
# serve module at CLI parse time).
LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


def _extract_bearer_token(auth_header: str) -> str | None:
    """Extract a bearer token from an Authorization header (Bearer scheme only)."""
    auth_header = (auth_header or "").strip()
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


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


async def _probe_rollout_health_ready(
    *,
    probe_url: str,
    max_attempts: int = 30,
    sleep_sec: float = 0.5,
    request_timeout_sec: float = 2.0,
    shutdown_event: Any | None = None,
) -> ProbeOutcome:
    """Poll GET ``probe_url`` until 200, exhaustion, or shutdown."""
    import asyncio

    import httpx

    async with httpx.AsyncClient() as client:
        for _ in range(max_attempts):
            if shutdown_event is not None and shutdown_event.is_set():
                return "shutdown"
            try:
                r = await client.get(probe_url, timeout=request_timeout_sec)
                if r.status_code == 200:
                    return "ready"
            except Exception:
                pass
            if shutdown_event is not None and shutdown_event.is_set():
                return "shutdown"
            if sleep_sec > 0:
                if shutdown_event is not None:
                    try:
                        await asyncio.wait_for(
                            shutdown_event.wait(),
                            timeout=sleep_sec,
                        )
                        return "shutdown"
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(sleep_sec)
    return "exhausted"


_REGISTRATION_TASK_AWAIT_TIMEOUT_SEC = 30.0


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
    from osmosis_ai.rollout_v2.backend.base import ExecutionBackend, ResultCallback
    from osmosis_ai.rollout_v2.types import ExecutionRequest, ExecutionResult

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
    skip_register: bool,
    local: bool,
    console: Console,
) -> None:
    from contextlib import asynccontextmanager

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

    if local and config.registration_api_key is not None:
        console.print_error(
            "Error: --local cannot be used when [registration].api_key is set in the "
            "config file."
        )
        raise typer.Exit(1)

    from osmosis_ai.eval.common.cli import auto_discover_grader, load_workflow

    workflow_cls, workflow_config, wf_err = load_workflow(
        rollout=config.serve_rollout,
        entrypoint=config.serve_entrypoint,
        quiet=False,
        console=console,
    )
    if wf_err or workflow_cls is None:
        console.print_error(wf_err or "Failed to load workflow.")
        raise typer.Exit(1)

    try:
        grader_cls, grader_config = auto_discover_grader(config.serve_entrypoint)
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
    do_register = not (skip_register or local or config.registration_skip)

    from osmosis_ai.rollout_v2.validator import validate_backend

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

    from osmosis_ai.rollout_v2.validator import resolved_agent_name

    agent_name = resolved_agent_name(workflow_cls, workflow_config)

    api_key: str | None
    if local:
        api_key = None
    else:
        from osmosis_ai.rollout.server.api_key import generate_api_key

        if config.registration_api_key is not None:
            api_key = config.registration_api_key
        else:
            api_key = generate_api_key()

    from osmosis_ai.rollout_v2.backend.local import LocalBackend

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

    credentials = None
    if do_register:
        from osmosis_ai.platform.auth.credentials import load_credentials

        credentials = load_credentials()
        if credentials is None:
            console.print_error(
                "Not logged in. Run 'osmosis auth login' first, or use "
                "--skip-register for local testing."
            )
            raise typer.Exit(1)

    registration_lifespan = None
    if do_register and credentials is not None:
        import asyncio

        from osmosis_ai.platform.cli.registration import (
            probe_host,
            register_with_platform,
        )

        @asynccontextmanager
        async def _lifespan(_app: object):
            shutdown_event = asyncio.Event()

            async def _probe_and_register() -> None:
                probe_target = probe_host(serve_host)
                probe_url = f"{_format_http_origin(probe_target, serve_port)}/health"
                outcome = await _probe_rollout_health_ready(
                    probe_url=probe_url,
                    shutdown_event=shutdown_event,
                )
                if outcome == "shutdown":
                    return
                if outcome == "exhausted":
                    console.print(
                        "Warning: /health probe did not return HTTP 200 before giving up; "
                        "attempting platform registration anyway."
                    )
                if shutdown_event.is_set():
                    return
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: register_with_platform(
                        host=serve_host,
                        port=serve_port,
                        agent_loop_name=agent_name,
                        credentials=credentials,
                        api_key=api_key,
                    ),
                )
                if result.success:
                    if result.is_healthy:
                        console.print(
                            f"Platform registration succeeded "
                            f"(server_id={result.server_id})."
                        )
                    else:
                        console.print(
                            f"Warning: registration completed but server status is not "
                            f"healthy: {result.error or result.status}"
                        )
                else:
                    console.print_error(
                        f"Platform registration failed: {result.error or 'unknown error'}"
                    )

            reg_task = asyncio.create_task(_probe_and_register())
            try:
                yield
            finally:
                shutdown_event.set()
                try:
                    await asyncio.wait_for(
                        reg_task,
                        timeout=_REGISTRATION_TASK_AWAIT_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    reg_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reg_task
                except Exception:
                    pass

        registration_lifespan = _lifespan

    from osmosis_ai.rollout_v2.server.app import create_rollout_server

    fastapi_app = create_rollout_server(backend=backend, lifespan=registration_lifespan)

    if not local:
        from fastapi import FastAPI

        assert isinstance(fastapi_app, FastAPI)

        @fastapi_app.get("/platform/health")
        async def platform_health() -> dict[str, object]:
            snap = backend.limiter.snapshot()
            return {
                "agent_loop": agent_name,
                "active_rollouts": snap.get("running", 0),
                "completed_rollouts": 0,
            }

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        from osmosis_ai.rollout.server.api_key import validate_api_key

        class _APIKeyAuthMiddleware(BaseHTTPMiddleware):
            def __init__(self, app: object, *, expected_key: str) -> None:
                super().__init__(app)
                self._expected_key = expected_key

            async def dispatch(self, request: Request, call_next):
                if request.method == "OPTIONS":
                    return await call_next(request)
                if request.method == "GET" and request.url.path == "/health":
                    return await call_next(request)
                auth_header = request.headers.get("authorization") or ""
                token = _extract_bearer_token(auth_header)
                if not validate_api_key(token, self._expected_key):
                    return _json_error_response(
                        status_code=401,
                        detail="Invalid or missing API key",
                    )
                return await call_next(request)

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
        fastapi_app.add_middleware(_APIKeyAuthMiddleware, expected_key=api_key)

    if local:
        api_key_display = "(disabled - local mode)"
    elif config.registration_api_key is not None:
        api_key_display = "(provided)"
    else:
        api_key_display = api_key or ""

    if local:
        reg_display = "Skipped (--local)"
    elif skip_register:
        reg_display = "Skipped (--skip-register)"
    elif config.registration_skip:
        reg_display = "Skipped ([registration].skip in config)"
    elif do_register:
        reg_display = "Enabled (after /health probe)"
    else:
        reg_display = "Skipped"

    grader_name = grader_cls.__name__ if grader_cls else "(none)"

    trace_display = (
        str(trace_session_path) if trace_session_path is not None else "(off)"
    )

    console.table(
        [
            ("Agent", agent_name),
            ("Address", _format_http_origin(serve_host, serve_port)),
            ("Grader", grader_name),
            ("API Key", api_key_display),
            ("Trace", trace_display),
            ("Registration", reg_display),
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
    skip_register: bool = typer.Option(
        False, "--skip-register", help="Skip registering with Platform."
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Local mode: no API key auth, no platform registration.",
    ),
) -> None:
    """Start a v2 RolloutServer from a TOML config file."""
    from osmosis_ai.cli.console import Console

    _serve(
        config_path=config_path,
        port=port,
        host=host,
        no_validate=no_validate,
        validate_only=validate_only,
        log_level=log_level,
        skip_register=skip_register,
        local=local,
        console=Console(),
    )


@app.command("test", hidden=True)
def test(
    module: str | None = typer.Option(
        None, "-m", "--module", "--agent", help="Module path."
    ),
    dataset: str = typer.Option(..., "-d", "--dataset", help="Path to dataset file."),
    model: str = typer.Option("gpt-5-mini", "--model", help="Model name."),
    limit: int | None = typer.Option(None, "--limit", help="Max rows."),
    offset: int = typer.Option(0, "--offset", help="Skip rows."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL."),
    debug: bool = typer.Option(False, "--debug", help="Debug output."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress output."),
) -> None:
    """Test an AgentWorkflow against a dataset (alias for eval run without grader)."""
    import tempfile
    from pathlib import Path

    from osmosis_ai.eval.evaluation.cli import EvalCommand

    if not module:
        from osmosis_ai.cli.console import Console

        Console().print_error("Error: --module (-m) is required.")
        raise typer.Exit(1)

    # Normalize model for LiteLLM
    llm_model = model if "/" in model else f"openai/{model}"

    # Shape matches osmosis eval run (rollout + entrypoint + dataset); rollout test is a thin wrapper.
    _ds = dataset.replace("\\", "/").replace('"', '\\"')
    toml_content = f'''[eval]
rollout = "_rollout_test"
entrypoint = "workflow.py"
dataset = "{_ds}"

[llm]
model = "{llm_model}"
'''
    if base_url:
        toml_content += f'base_url = "{base_url}"\n'

    # --api-key → write to temp env var, reference in TOML via api_key_env
    _tmp_env_key = "_OSMOSIS_EVAL_TMP_API_KEY"
    if api_key:
        import os

        os.environ[_tmp_env_key] = api_key
        toml_content += f'api_key_env = "{_tmp_env_key}"\n'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        tmp_path = f.name

    try:
        cmd = EvalCommand()
        rc = cmd.run(
            config_path=tmp_path,
            fresh=False,
            retry_failed=False,
            limit=limit,
            offset=offset,
            quiet=quiet,
            debug=debug,
            output_path=None,
            log_samples=False,
            batch_size_override=None,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up temp env var
        if api_key:
            import os

            os.environ.pop(_tmp_env_key, None)

    if rc:
        raise typer.Exit(rc)


@app.command("list")
def list_rollouts() -> None:
    """List registered rollouts."""
    not_implemented("rollout", "list")
