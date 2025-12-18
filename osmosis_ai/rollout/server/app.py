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
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

from osmosis_ai.rollout._compat import FASTAPI_AVAILABLE
from osmosis_ai.rollout.config.settings import RolloutSettings, get_settings
from osmosis_ai.rollout.core.base import RolloutAgentLoop, RolloutContext
from osmosis_ai.rollout.core.schemas import InitResponse, RolloutRequest
from osmosis_ai.rollout.server.state import AppState
from osmosis_ai.rollout.client import OsmosisLLMClient

logger = logging.getLogger(__name__)


def create_app(
    agent_loop: RolloutAgentLoop,
    max_concurrent: Optional[int] = None,
    record_ttl_seconds: Optional[float] = None,
    settings: Optional[RolloutSettings] = None,
) -> "FastAPI":
    """Create a FastAPI application for the agent loop.

    This factory creates a complete FastAPI application with:
    - POST /v1/rollout/init: Accept rollout requests (returns 202 Accepted)
    - GET /health: Health check endpoint
    - Background task management with concurrency control
    - Idempotency handling (duplicate requests return same response)
    - Automatic cleanup of completed rollout records

    Args:
        agent_loop: The RolloutAgentLoop implementation to use.
        max_concurrent: Maximum concurrent rollouts. Defaults to settings.
        record_ttl_seconds: TTL for completed records. Defaults to settings.
        settings: Configuration settings. Defaults to global settings.

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
        state.start_cleanup_task()
        yield
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

    @app.post("/v1/rollout/init", status_code=202)
    async def init_rollout(request: RolloutRequest) -> InitResponse:
        """Initialize a new rollout.

        This endpoint accepts a rollout request and starts the agent loop
        in the background. Returns 202 Accepted immediately with the tools
        available for this rollout.

        Idempotency: If a rollout with the same ID is already running or
        recently completed, returns the same tools without starting a new rollout.
        """
        init_future, created = state.get_or_create_init_future(request.rollout_id)

        # Duplicate request: await the same InitResponse and return it.
        if not created:
            logger.debug("Duplicate rollout request: rollout_id=%s", request.rollout_id)
            init_response = await init_future
            return init_response

        try:
            # Leader request: compute tools once and cache InitResponse.
            tools = agent_loop.get_tools(request)

            # Define the background task
            async def run_rollout() -> None:
                """Execute the rollout in the background."""
                start_time = time.monotonic()

                async with state.semaphore:
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

                                await llm.complete_rollout(
                                    status=result.status,
                                    final_messages=result.final_messages,
                                    finish_reason=result.finish_reason,
                                    error_message=result.error_message,
                                    metrics=result.metrics,
                                )

                                duration = time.monotonic() - start_time

                                logger.info(
                                    "Rollout completed: rollout_id=%s, status=%s, "
                                    "finish_reason=%s, duration=%.2fs",
                                    request.rollout_id,
                                    result.status,
                                    result.finish_reason,
                                    duration,
                                )

                            except Exception as e:
                                # Agent loop error
                                logger.error(
                                    "Rollout agent error: rollout_id=%s, error=%s",
                                    request.rollout_id,
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
                            request.rollout_id,
                            str(e),
                            exc_info=True,
                        )

                    finally:
                        state.mark_completed(request.rollout_id)

            # Start background task
            task = asyncio.create_task(run_rollout())
            state.mark_started(request.rollout_id, task)

            init_response = InitResponse(rollout_id=request.rollout_id, tools=tools)
            init_future.set_result(init_response)

            logger.info(
                "Rollout started: rollout_id=%s, tool_count=%d",
                request.rollout_id,
                len(tools),
            )

            return init_response
        except Exception as e:
            if not init_future.done():
                init_future.set_exception(e)
            state.clear_init_record(request.rollout_id)
            raise

    return app
