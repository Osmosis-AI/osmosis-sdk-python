"""FastAPI application factory for RolloutAgentLoop implementations.

This module provides the create_app() factory function that creates
a complete FastAPI application for serving RolloutAgentLoop implementations.

Example:
    from osmosis_ai.rollout.server import create_app
    from my_agent import MyAgentLoop

    app = create_app(MyAgentLoop())

    # Run with: uvicorn main:app --port 9000
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI, HTTPException, Request

    from osmosis_ai.auth.credentials import WorkspaceCredentials

from osmosis_ai.rollout._compat import FASTAPI_AVAILABLE
from osmosis_ai.rollout.client import OsmosisLLMClient
from osmosis_ai.rollout.config.settings import RolloutSettings, get_settings
from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext
from osmosis_ai.rollout.core.schemas import InitResponse, RolloutRequest
from osmosis_ai.rollout.server.api_key import validate_api_key
from osmosis_ai.rollout.server.state import AppState

logger = logging.getLogger(__name__)

# NOTE: FastAPI is an optional dependency. We avoid importing it at module import
# time unless it's available, but we DO need these symbols in module globals so
# FastAPI can resolve forward-referenced annotations (due to postponed eval).
if FASTAPI_AVAILABLE:
    from fastapi import HTTPException, Request


def _extract_bearer_token(auth_header: str) -> str | None:
    """Extract a bearer token from an Authorization header.

    Accepts both:
    - "Bearer <token>"
    - "<token>" (raw token fallback)
    """
    auth_header = (auth_header or "").strip()
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return auth_header


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """Manages application startup, shutdown, and platform registration."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        state: AppState,
        *,
        debug_dir: str | None = None,
        credentials: WorkspaceCredentials | None = None,
        server_host: str | None = None,
        server_port: int | None = None,
        api_key: str | None = None,
        on_startup: Callable[[], Awaitable[None]] | None = None,
        on_shutdown: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._state = state
        self._debug_dir = debug_dir
        self._credentials = credentials
        self._server_host = server_host
        self._server_port = server_port
        self._api_key = api_key
        self._on_startup = on_startup
        self._on_shutdown = on_shutdown

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle."""
        logger.info(
            "Server starting: agent_loop=%s, max_concurrent=%d",
            self._agent_loop.name,
            self._state._max_concurrent,
        )
        if self._debug_dir:
            logger.info("Debug logging enabled: output_dir=%s", self._debug_dir)
        self._state.start_cleanup_task()

        if self._on_startup is not None:
            await self._on_startup()

        # Start platform registration as a background task.
        # The task waits briefly for the server to be ready (after yield),
        # then registers with Platform. This ensures the health check succeeds.
        registration_task = self._start_registration()

        try:
            yield
        finally:
            await self._await_registration(registration_task)

            if self._on_shutdown is not None:
                await self._on_shutdown()

            logger.info("Server stopping")
            await self._state.stop_cleanup_task()
            await self._state.cancel_all()

    # -- Registration helpers ------------------------------------------------

    def _start_registration(self) -> asyncio.Task | None:
        """Start platform registration as a background task, if configured."""
        if (
            self._credentials is not None
            and self._server_host is not None
            and self._server_port is not None
        ):
            return asyncio.create_task(self._do_registration())
        return None

    async def _await_registration(self, task: asyncio.Task | None) -> None:
        """Wait for a registration task to finish, suppressing errors."""
        if task is None:
            return
        try:
            await asyncio.wait_for(
                task,
                timeout=self._state.settings.registration_shutdown_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Platform registration timed out")
        except asyncio.CancelledError:
            logger.warning("Platform registration was cancelled")
        except Exception as e:
            logger.error("Platform registration failed: %s", e)

    async def _do_registration(self) -> None:
        """Poll for server readiness, then register with Platform."""
        from osmosis_ai.rollout.server.registration import (
            print_registration_result,
            register_with_platform,
        )

        # Guaranteed non-None by _start_registration() guard
        assert self._server_host is not None
        assert self._server_port is not None
        assert self._credentials is not None
        host: str = self._server_host
        port: int = self._server_port
        credentials: WorkspaceCredentials = self._credentials

        await self._wait_for_server_ready(port)

        # Run sync registration in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(
            register_with_platform,
            host=host,
            port=port,
            agent_loop_name=self._agent_loop.name,
            credentials=credentials,
            api_key=self._api_key,
        )
        print_registration_result(
            result=result,
            host=host,
            port=port,
            agent_loop_name=self._agent_loop.name,
            api_key=self._api_key,
        )

    async def _wait_for_server_ready(self, port: int) -> None:
        """Poll the local health endpoint until the server is accepting requests."""
        import httpx

        poll_interval = (
            self._state.settings.registration_readiness_poll_interval_seconds
        )
        timeout = self._state.settings.registration_readiness_timeout_seconds
        health_url = f"http://127.0.0.1:{port}/health"

        start_time = time.monotonic()
        server_ready = False

        async with httpx.AsyncClient() as client:
            while time.monotonic() - start_time < timeout:
                try:
                    resp = await client.get(health_url, timeout=1.0)
                    if resp.status_code == 200:
                        server_ready = True
                        break
                except httpx.ConnectError:
                    # Server not listening yet, expected during startup
                    pass
                except httpx.RequestError as e:
                    # Other request errors (timeout, etc.)
                    logger.debug("Health check failed: %s", e)
                await asyncio.sleep(poll_interval)

        elapsed = time.monotonic() - start_time
        if server_ready:
            logger.debug("Server ready for registration in %.2fs", elapsed)
        else:
            logger.warning(
                "Server did not become ready within %.1fs, "
                "attempting registration anyway",
                timeout,
            )


# ---------------------------------------------------------------------------
# RolloutExecutor
# ---------------------------------------------------------------------------


class RolloutExecutor:
    """Handles rollout initialization and background execution."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        state: AppState,
        *,
        debug_dir: str | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._state = state
        self._debug_dir = debug_dir

    async def init(self, rollout_request: RolloutRequest) -> InitResponse:
        """Initialize a rollout: check idempotency, compute tools, start execution.

        Returns the InitResponse immediately; actual execution runs in the background.
        """
        key = rollout_request.idempotency_key or rollout_request.rollout_id
        init_future, created = self._state.get_or_create_init_future(key)

        # Duplicate request: await the same InitResponse and return it.
        if not created:
            logger.debug(
                "Duplicate rollout request: rollout_id=%s",
                rollout_request.rollout_id,
            )
            return await init_future

        task: asyncio.Task | None = None
        try:
            # Leader request: compute tools once and cache InitResponse.
            tools = self._agent_loop.get_tools(rollout_request)

            task = asyncio.create_task(self._execute(rollout_request, tools, key))
            self._state.mark_started(key, task)

            init_response = InitResponse(
                rollout_id=rollout_request.rollout_id, tools=tools
            )
            init_future.set_result(init_response)

            logger.info(
                "Rollout started: rollout_id=%s, tool_count=%d",
                rollout_request.rollout_id,
                len(tools),
            )

            return init_response
        except Exception as e:
            # Cancel the background task if it was already created
            if task is not None:
                task.cancel()
            if not init_future.done():
                init_future.set_exception(e)
            self._state.clear_init_record(key)
            raise

    async def _execute(
        self,
        rollout_request: RolloutRequest,
        tools: list,
        key: str,
    ) -> None:
        """Run the rollout as a background task with concurrency control."""
        start_time = time.monotonic()

        try:
            async with self._state.semaphore:
                try:
                    await self._run_with_client(rollout_request, tools, start_time)
                except Exception as e:
                    # Client/infrastructure error
                    logger.error(
                        "Rollout infrastructure error: rollout_id=%s, error=%s",
                        rollout_request.rollout_id,
                        str(e),
                        exc_info=True,
                    )
        finally:
            self._state.mark_completed(key)

    async def _run_with_client(
        self,
        rollout_request: RolloutRequest,
        tools: list,
        start_time: float,
    ) -> None:
        """Open an LLM client connection and run the agent loop."""
        async with OsmosisLLMClient(
            server_url=rollout_request.server_url,
            rollout_id=rollout_request.rollout_id,
            api_key=rollout_request.api_key,
        ) as llm:
            ctx = RolloutContext(
                request=rollout_request,
                tools=tools,
                llm=llm,
                _start_time=start_time,
                _debug_dir=self._debug_dir,
            )

            try:
                result = await self._agent_loop.run(ctx)

                await llm.complete_rollout(
                    status=result.status,
                    final_messages=result.final_messages,
                    finish_reason=result.finish_reason,
                    error_message=result.error_message,
                    metrics=result.metrics,
                    reward=result.reward,
                )

                duration = time.monotonic() - start_time

                logger.info(
                    "Rollout completed: rollout_id=%s, status=%s, "
                    "finish_reason=%s, duration=%.2fs",
                    rollout_request.rollout_id,
                    result.status,
                    result.finish_reason,
                    duration,
                )

            except Exception as e:
                # Agent loop error
                logger.error(
                    "Rollout agent error: rollout_id=%s, error=%s",
                    rollout_request.rollout_id,
                    str(e),
                    exc_info=True,
                )

                await llm.complete_rollout(
                    status="ERROR",
                    final_messages=[],
                    finish_reason="error",
                    error_message=str(e),
                )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(
    app: FastAPI,
    *,
    agent_loop: RolloutAgentLoop,
    state: AppState,
    executor: RolloutExecutor,
    api_key: str | None,
) -> None:
    """Register all API routes on the FastAPI application."""

    def _health_status() -> dict[str, Any]:
        return {
            "status": "healthy",
            "agent_loop": agent_loop.name,
            "active_rollouts": state.active_count,
            "completed_rollouts": state.completed_count,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint.

        Returns server status and statistics.
        """
        return _health_status()

    @app.get("/platform/health")
    async def platform_health(request: Request) -> dict[str, Any]:
        """Platform health check endpoint (authenticated).

        This endpoint is intended for Osmosis Platform to validate:
        - Reachability of the server
        - Correctness of the configured RolloutServer API key

        It requires: Authorization: Bearer <api_key>
        """
        # If API key auth is disabled (e.g., local_debug), do not expose this endpoint.
        if api_key is None:
            raise HTTPException(status_code=404, detail="Not found")

        provided = _extract_bearer_token(request.headers.get("authorization") or "")

        if not validate_api_key(provided, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return _health_status()

    @app.post("/v1/rollout/init", status_code=202)
    async def init_rollout(
        rollout_request: RolloutRequest,
        http_request: Request,
    ) -> InitResponse:
        """Initialize a new rollout.

        This endpoint accepts a rollout request and starts the agent loop
        in the background. Returns 202 Accepted immediately with the tools
        available for this rollout.

        Idempotency: If a rollout with the same ID is already running or
        recently completed, returns the same tools without starting a new rollout.
        """
        # Validate RolloutServer auth if configured:
        # TrainGate must send: Authorization: Bearer <api_key>
        if api_key is not None:
            provided = _extract_bearer_token(
                http_request.headers.get("authorization") or ""
            )
            if not validate_api_key(provided, api_key):
                logger.warning(
                    "Invalid API key for rollout request: rollout_id=%s",
                    rollout_request.rollout_id,
                )
                raise HTTPException(
                    status_code=401, detail="Invalid or missing API key"
                )

        return await executor.init(rollout_request)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop: RolloutAgentLoop,
    max_concurrent: int | None = None,
    record_ttl_seconds: float | None = None,
    settings: RolloutSettings | None = None,
    credentials: WorkspaceCredentials | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
    api_key: str | None = None,
    debug_dir: str | None = None,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Create a FastAPI application for the agent loop.

    This factory creates a complete FastAPI application with:
    - POST /v1/rollout/init: Accept rollout requests (returns 202 Accepted)
    - GET /health: Health check endpoint
    - Background task management with concurrency control
    - Idempotency handling (duplicate requests return same response)
    - Automatic cleanup of completed rollout records
    - Optional platform registration on startup
    - API key authentication for incoming requests

    Args:
        agent_loop: The RolloutAgentLoop implementation to use.
        max_concurrent: Maximum concurrent rollouts. Defaults to settings.
        record_ttl_seconds: TTL for completed records. Defaults to settings.
        settings: Configuration settings. Defaults to global settings.
        credentials: Workspace credentials for platform registration.
                     If None, registration is skipped.
        server_host: Host the server is bound to (for registration).
        server_port: Port the server is listening on (for registration).
        api_key: API key for authenticating incoming requests.
                 If provided, requests must include:
                 - Authorization: Bearer <api_key>
        debug_dir: Optional directory for debug logging.
                   If provided, each rollout will write detailed execution
                   traces to {debug_dir}/{rollout_id}.jsonl files.
                   Disabled by default.
        on_startup: Optional async callback to run during application startup.
                    Use this for custom initialization (e.g., warming caches,
                    starting background services).
        on_shutdown: Optional async callback to run during application shutdown.
                     Use this for custom cleanup (e.g., stopping services,
                     releasing resources).

    Returns:
        FastAPI application ready to serve.

    Raises:
        ImportError: If FastAPI is not installed.

    Example:
        from my_agent import MyAgentLoop

        app = create_app(MyAgentLoop())

        # Run with: uvicorn main:app --port 9000

        # Or with custom settings:
        from osmosis_ai.rollout.config import RolloutSettings, RolloutServerSettings

        app = create_app(
            MyAgentLoop(),
            settings=RolloutSettings(
                server=RolloutServerSettings(max_concurrent_rollouts=200),
            ),
        )
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for create_app(). "
            "Install it with: pip install osmosis-ai[server]"
        )

    from fastapi import FastAPI

    if settings is None:
        settings = get_settings()

    state = AppState(
        max_concurrent=max_concurrent,
        record_ttl_seconds=record_ttl_seconds,
        settings=settings.server,
        agent_loop_name=agent_loop.name,
    )

    lifecycle = LifecycleManager(
        agent_loop,
        state,
        debug_dir=debug_dir,
        credentials=credentials,
        server_host=server_host,
        server_port=server_port,
        api_key=api_key,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )

    executor = RolloutExecutor(
        agent_loop,
        state,
        debug_dir=debug_dir,
    )

    app = FastAPI(
        title=f"Osmosis RolloutServer ({agent_loop.name})",
        description="Remote rollout server for Osmosis agent training",
        lifespan=lifecycle.lifespan,
    )

    _register_routes(
        app,
        agent_loop=agent_loop,
        state=state,
        executor=executor,
        api_key=api_key,
    )

    return app
