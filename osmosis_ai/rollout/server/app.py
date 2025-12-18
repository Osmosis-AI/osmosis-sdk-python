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
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

from osmosis_ai.rollout._compat import FASTAPI_AVAILABLE
from osmosis_ai.rollout.config.settings import RolloutSettings, get_settings
from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext
from osmosis_ai.rollout.core.schemas import InitResponse, RolloutRequest
from osmosis_ai.rollout.observability.logging import (
    configure_logging,
    get_logger,
    set_rollout_id,
    clear_context,
)
from osmosis_ai.rollout.observability.metrics import (
    configure_metrics,
    get_metrics,
    get_metrics_content,
)
from osmosis_ai.rollout.observability.tracing import configure_tracing, span, SpanNames
from osmosis_ai.rollout.server.middleware import add_observability_middleware
from osmosis_ai.rollout.server.state import AppState
from osmosis_ai.rollout.client import OsmosisLLMClient

logger = get_logger(__name__)


def create_app(
    agent_loop: RolloutAgentLoop,
    max_concurrent: Optional[int] = None,
    record_ttl_seconds: Optional[float] = None,
    settings: Optional[RolloutSettings] = None,
    enable_metrics_endpoint: bool = True,
) -> "FastAPI":
    """Create a FastAPI application for the agent loop.

    This factory creates a complete FastAPI application with:
    - POST /v1/rollout/init: Accept rollout requests (returns 202 Accepted)
    - GET /health: Health check endpoint
    - GET /metrics: Prometheus metrics endpoint (when enabled)
    - Background task management with concurrency control
    - Idempotency handling (duplicate requests return same response)
    - Automatic cleanup of completed rollout records
    - Integrated logging, tracing, and metrics

    Args:
        agent_loop: The RolloutAgentLoop implementation to use.
        max_concurrent: Maximum concurrent rollouts. Defaults to settings.
        record_ttl_seconds: TTL for completed records. Defaults to settings.
        settings: Configuration settings. Defaults to global settings.
        enable_metrics_endpoint: Whether to expose /metrics endpoint.

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
    from starlette.responses import Response

    # Load settings
    if settings is None:
        settings = get_settings()

    # Configure observability
    configure_logging(settings.logging)
    configure_tracing(settings.tracing)
    configure_metrics(settings.metrics)

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
            "server_starting",
            agent_loop=agent_loop.name,
            max_concurrent=state._max_concurrent,
        )
        state.start_cleanup_task()
        yield
        logger.info("server_stopping")
        await state.stop_cleanup_task()
        await state.cancel_all()

    app = FastAPI(
        title=f"Osmosis RolloutServer ({agent_loop.name})",
        description="Remote rollout server for Osmosis agent training",
        lifespan=lifespan,
    )

    # Add observability middleware
    add_observability_middleware(app)

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

    # Add metrics endpoint if enabled
    if enable_metrics_endpoint and settings.metrics.enabled and settings.metrics.expose_endpoint:
        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            """Prometheus metrics endpoint."""
            content, content_type = get_metrics_content()
            return Response(content=content, media_type=content_type)

    @app.post("/v1/rollout/init", status_code=202)
    async def init_rollout(request: RolloutRequest) -> InitResponse:
        """Initialize a new rollout.

        This endpoint accepts a rollout request and starts the agent loop
        in the background. Returns 202 Accepted immediately with the tools
        available for this rollout.

        Idempotency: If a rollout with the same ID is already running or
        recently completed, returns the same tools without starting a new rollout.
        """
        metrics = get_metrics()

        with span(
            SpanNames.ROLLOUT_INIT,
            attributes={
                "rollout_id": request.rollout_id,
                "agent_loop": agent_loop.name,
            },
        ) as s:
            init_future, created = state.get_or_create_init_future(request.rollout_id)

            # Duplicate request: await the same InitResponse and return it.
            if not created:
                logger.debug(
                    "duplicate_rollout_request",
                    rollout_id=request.rollout_id,
                )
                s.set_attribute("duplicate", True)
                init_response = await init_future
                s.set_attribute("tool_count", len(init_response.tools))
                return init_response

            try:
                # Leader request: compute tools once and cache InitResponse.
                tools = agent_loop.get_tools(request)
                s.set_attribute("tool_count", len(tools))

                # Define the background task
                async def run_rollout() -> None:
                    """Execute the rollout in the background."""
                    set_rollout_id(request.rollout_id)
                    start_time = time.monotonic()

                    async with state.semaphore:
                        with span(
                            SpanNames.ROLLOUT_RUN,
                            attributes={
                                "rollout_id": request.rollout_id,
                                "max_turns": request.max_turns,
                            },
                        ) as run_span:
                            try:
                                async with OsmosisLLMClient(
                                    server_url=request.server_url,
                                    rollout_id=request.rollout_id,
                                    api_key=request.api_key,
                                ) as llm:
                                    ctx = RolloutContext(
                                        request=request,
                                        tools=tools,
                                        llm=llm,
                                        _start_time=start_time,
                                    )

                                    try:
                                        result = await agent_loop.run(ctx)

                                        with span(SpanNames.ROLLOUT_COMPLETE):
                                            await llm.complete_rollout(
                                                status=result.status,
                                                final_messages=result.final_messages,
                                                finish_reason=result.finish_reason,
                                                error_message=result.error_message,
                                                metrics=result.metrics,
                                            )

                                        # Record success metrics
                                        duration = time.monotonic() - start_time
                                        metrics.rollouts_completed.labels(
                                            agent_loop=agent_loop.name,
                                            status=result.status,
                                            finish_reason=result.finish_reason,
                                        ).inc()
                                        metrics.rollout_duration_seconds.labels(
                                            agent_loop=agent_loop.name,
                                            status=result.status,
                                        ).observe(duration)

                                        run_span.set_attribute("status", result.status)
                                        run_span.set_attribute(
                                            "finish_reason", result.finish_reason
                                        )

                                        logger.info(
                                            "rollout_completed",
                                            rollout_id=request.rollout_id,
                                            status=result.status,
                                            finish_reason=result.finish_reason,
                                            duration_seconds=duration,
                                        )

                                    except Exception as e:
                                        # Agent loop error
                                        logger.error(
                                            "rollout_agent_error",
                                            rollout_id=request.rollout_id,
                                            error=str(e),
                                            exc_info=True,
                                        )
                                        run_span.set_attribute("status", "ERROR")
                                        run_span.set_attribute("error", str(e))

                                        with span(SpanNames.ROLLOUT_ERROR):
                                            await llm.complete_rollout(
                                                status="ERROR",
                                                final_messages=[],
                                                finish_reason="error",
                                                error_message=str(e),
                                            )

                                        # Record error metrics
                                        metrics.rollouts_failed.labels(
                                            agent_loop=agent_loop.name,
                                            error_type=type(e).__name__,
                                        ).inc()

                            except Exception as e:
                                # Client/infrastructure error
                                logger.error(
                                    "rollout_infrastructure_error",
                                    rollout_id=request.rollout_id,
                                    error=str(e),
                                    exc_info=True,
                                )
                                metrics.rollouts_failed.labels(
                                    agent_loop=agent_loop.name,
                                    error_type=type(e).__name__,
                                ).inc()

                            finally:
                                state.mark_completed(request.rollout_id)
                                clear_context()

                # Start background task
                task = asyncio.create_task(run_rollout())
                state.mark_started(request.rollout_id, task)

                # Record metrics (once per rollout)
                metrics.rollouts_started.labels(agent_loop=agent_loop.name).inc()

                init_response = InitResponse(rollout_id=request.rollout_id, tools=tools)
                init_future.set_result(init_response)

                logger.info(
                    "rollout_started",
                    rollout_id=request.rollout_id,
                    tool_count=len(tools),
                )

                return init_response
            except Exception as e:
                if not init_future.done():
                    init_future.set_exception(e)
                state.clear_init_record(request.rollout_id)
                raise

    return app
