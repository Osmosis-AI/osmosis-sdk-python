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
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI, HTTPException, Request
    from osmosis_ai.auth.credentials import WorkspaceCredentials

from osmosis_ai.rollout._compat import FASTAPI_AVAILABLE
from osmosis_ai.rollout.config.settings import RolloutSettings, get_settings
from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext
from osmosis_ai.rollout.core.schemas import InitResponse, RolloutRequest
from osmosis_ai.rollout.server.api_key import validate_api_key
from osmosis_ai.rollout.server.state import AppState
from osmosis_ai.rollout.client import OsmosisLLMClient

logger = logging.getLogger(__name__)

# NOTE: FastAPI is an optional dependency. We avoid importing it at module import
# time unless it's available, but we DO need these symbols in module globals so
# FastAPI can resolve forward-referenced annotations (due to postponed eval).
if FASTAPI_AVAILABLE:
    from fastapi import HTTPException, Request


def _extract_bearer_token(auth_header: str) -> Optional[str]:
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


def create_app(
    agent_loop: RolloutAgentLoop,
    max_concurrent: Optional[int] = None,
    record_ttl_seconds: Optional[float] = None,
    settings: Optional[RolloutSettings] = None,
    credentials: Optional["WorkspaceCredentials"] = None,
    server_host: Optional[str] = None,
    server_port: Optional[int] = None,
    api_key: Optional[str] = None,
    debug_dir: Optional[str] = None,
    on_startup: Optional[Callable[[], Awaitable[None]]] = None,
    on_shutdown: Optional[Callable[[], Awaitable[None]]] = None,
) -> "FastAPI":
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

    # Load settings
    if settings is None:
        settings = get_settings()

    # Create app state
    state = AppState(
        max_concurrent=max_concurrent,
        record_ttl_seconds=record_ttl_seconds,
        settings=settings.server,
        agent_loop_name=agent_loop.name,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        logger.info(
            "Server starting: agent_loop=%s, max_concurrent=%d",
            agent_loop.name,
            state._max_concurrent,
        )
        if debug_dir:
            logger.info("Debug logging enabled: output_dir=%s", debug_dir)
        state.start_cleanup_task()

        # Run custom startup callback
        if on_startup is not None:
            await on_startup()

        # Start platform registration as a background task.
        # The task waits briefly for the server to be ready (after yield),
        # then registers with Platform. This ensures the health check succeeds.
        registration_task: Optional[asyncio.Task] = None
        if credentials is not None and server_host is not None and server_port is not None:
            from osmosis_ai.rollout.server.registration import (
                register_with_platform,
                print_registration_result,
            )

            async def do_registration():
                import httpx

                # Poll health endpoint until server is ready
                poll_interval = state.settings.registration_readiness_poll_interval_seconds
                timeout = state.settings.registration_readiness_timeout_seconds
                health_url = f"http://127.0.0.1:{server_port}/health"

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
                        "Server did not become ready within %.1fs, attempting registration anyway",
                        timeout,
                    )

                # Run sync registration in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(
                    register_with_platform,
                    host=server_host,
                    port=server_port,
                    agent_loop_name=agent_loop.name,
                    credentials=credentials,
                    api_key=api_key,
                )
                print_registration_result(
                    result=result,
                    host=server_host,
                    port=server_port,
                    agent_loop_name=agent_loop.name,
                    api_key=api_key,
                )

            registration_task = asyncio.create_task(do_registration())

        yield

        # Wait for registration to complete before shutdown
        if registration_task is not None:
            try:
                await asyncio.wait_for(
                    registration_task,
                    timeout=state.settings.registration_shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Platform registration timed out")
            except asyncio.CancelledError:
                logger.warning("Platform registration was cancelled")
            except Exception as e:
                logger.error("Platform registration failed: %s", e)

        # Run custom shutdown callback
        if on_shutdown is not None:
            await on_shutdown()

        logger.info("Server stopping")
        await state.stop_cleanup_task()
        await state.cancel_all()

    app = FastAPI(
        title=f"Osmosis RolloutServer ({agent_loop.name})",
        description="Remote rollout server for Osmosis agent training",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint.

        Returns server status and statistics.
        """
        return {
            "status": "healthy",
            "agent_loop": agent_loop.name,
            "active_rollouts": state.active_count,
            "completed_rollouts": state.completed_count,
        }

    @app.get("/platform/health")
    async def platform_health(request: Request) -> Dict[str, Any]:
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

        return {
            "status": "healthy",
            "agent_loop": agent_loop.name,
            "active_rollouts": state.active_count,
            "completed_rollouts": state.completed_count,
        }

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
            provided = _extract_bearer_token(http_request.headers.get("authorization") or "")
            if not validate_api_key(provided, api_key):
                logger.warning(
                    "Invalid API key for rollout request: rollout_id=%s",
                    rollout_request.rollout_id,
                )
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        # Idempotency key: prefer request.idempotency_key, fallback to rollout_id.
        key = rollout_request.idempotency_key or rollout_request.rollout_id

        init_future, created = state.get_or_create_init_future(key)

        # Duplicate request: await the same InitResponse and return it.
        if not created:
            logger.debug(
                "Duplicate rollout request: rollout_id=%s",
                rollout_request.rollout_id,
            )
            init_response = await init_future
            return init_response

        try:
            # Leader request: compute tools once and cache InitResponse.
            tools = agent_loop.get_tools(rollout_request)

            # Define the background task
            async def run_rollout() -> None:
                """Execute the rollout in the background."""
                start_time = time.monotonic()

                async with state.semaphore:
                    try:
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
                                _debug_dir=debug_dir,
                            )

                            try:
                                result = await agent_loop.run(ctx)

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

                    except Exception as e:
                        # Client/infrastructure error
                        logger.error(
                            "Rollout infrastructure error: rollout_id=%s, error=%s",
                            rollout_request.rollout_id,
                            str(e),
                            exc_info=True,
                        )

                    finally:
                        state.mark_completed(key)

            # Start background task
            task = asyncio.create_task(run_rollout())
            state.mark_started(key, task)

            init_response = InitResponse(rollout_id=rollout_request.rollout_id, tools=tools)
            init_future.set_result(init_response)

            logger.info(
                "Rollout started: rollout_id=%s, tool_count=%d",
                rollout_request.rollout_id,
                len(tools),
            )

            return init_response
        except Exception as e:
            if not init_future.done():
                init_future.set_exception(e)
            state.clear_init_record(key)
            raise

    return app
