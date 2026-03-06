import logging
import traceback
from typing import Any, Dict, Type

from fastapi import BackgroundTasks, FastAPI, HTTPException

from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.server_state import ConcurrencyLimiter, RolloutServerState
from osmosis_ai.rollout_v2.context import RolloutContext, GraderContext
from osmosis_ai.rollout_v2.utils.http_utils import post_json_with_retry
from osmosis_ai.rollout_v2.utils.misc import map_initial_messages_to_content_blocks
from osmosis_ai.rollout_v2.agent_workflow import (
    AgentWorkflow,
    AgentWorkflowContext,
)
from osmosis_ai.rollout_v2.types import (
    GraderStatus,
    RolloutErrorCategory,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutServerConfig,
    RolloutStatus,
    GraderInitRequest,
    GraderInitResponse,
    AgentWorkflowConfig,
)

logger = logging.getLogger(__name__)

def _categorize_exception(exc: Exception) -> RolloutErrorCategory:
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    if isinstance(exc, HTTPException):
        return RolloutErrorCategory.HTTP_ERROR
    return RolloutErrorCategory.AGENT_ERROR


async def run_grader_with_callback(
    grader: Grader,
    ctx: GraderContext,
    limiter: ConcurrencyLimiter,
) -> None:
    async with limiter.acquire():
        try:
            await grader.grade(ctx)

            graded_samples = ctx.get_samples()
            assert all(sample.reward is not None for sample in graded_samples.values()), "All samples must have a reward"
            ctx.set_grader_status(GraderStatus.SUCCESS)
        except Exception as e:
            logging.error(traceback.format_exc())
            ctx.set_grader_status(GraderStatus.FAILURE)
            ctx.set_grader_error(
                message=str(e),
                category=_categorize_exception(e),
            )

        # Run callback to notify the rollout server that the grader is complete.
        await post_json_with_retry(
            url=ctx.completion_callback_url,
            payload=ctx.make_grader_complete_request().model_dump(),
        )


async def run_agent_workflow_with_callback(
    agent_workflow: AgentWorkflow,
    agent_workflow_ctx: AgentWorkflowContext,
    rollout_ctx: RolloutContext,
    limiter: ConcurrencyLimiter,
) -> None:
    async with limiter.acquire():
        # Run the agent and set the final status
        with rollout_ctx:
            try:
                await agent_workflow.run(agent_workflow_ctx)
                rollout_ctx.set_rollout_status(RolloutStatus.SUCCESS)
            except Exception as e:
                logging.error(traceback.format_exc())
                rollout_ctx.set_rollout_status(RolloutStatus.FAILURE)
                rollout_ctx.set_rollout_error(
                    message=str(e),
                    category=_categorize_exception(e),
                )

        # Run callback to notify the rollout server that the rollout is complete.
        await post_json_with_retry(
            url=rollout_ctx.completion_callback_url,
            payload=rollout_ctx.make_rollout_complete_request().model_dump(),
        )

def create_app(
    *,
    agent_workflow_cls: Type[AgentWorkflow],
    grader_cls: Type[Grader],
    agent_workflow_config: AgentWorkflowConfig,
    server_config: RolloutServerConfig | None = None,
) -> FastAPI:
    rollout_server_state = RolloutServerState(server_config or RolloutServerConfig())
    app = FastAPI()
    app.state.rollout_server_state = rollout_server_state

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return rollout_server_state.health_payload()

    @app.post("/rollout")
    async def rollout(
        request: RolloutInitRequest, background_tasks: BackgroundTasks
    ) -> RolloutInitResponse:
        try:

            rollout_ctx = RolloutContext(
                rollout_id=request.rollout_id, 
                completion_callback_url=request.completion_callback_url,
                chat_completions_url=request.chat_completions_url,
            )
            agent_workflow = agent_workflow_cls(agent_workflow_config)
            agent_workflow_ctx = AgentWorkflowContext(
                prompt=map_initial_messages_to_content_blocks(request.initial_messages), 
                config=agent_workflow_config
            )
            
            background_tasks.add_task(
                run_agent_workflow_with_callback,
                agent_workflow,
                agent_workflow_ctx,
                rollout_ctx,
                rollout_server_state.agent_workflow_concurrency_limiter,
            )
            return RolloutInitResponse()
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/grader")
    async def grade(
        request: GraderInitRequest, background_tasks: BackgroundTasks
    ) -> GraderInitResponse:
        try:
            grader_ctx = GraderContext(rollout_id=request.rollout_id, 
                completion_callback_url=request.completion_callback_url, 
                samples=request.samples,
            )
            grader = grader_cls()
            background_tasks.add_task(
                run_grader_with_callback,
                grader,
                grader_ctx,
                rollout_server_state.grader_concurrency_limiter,
            )
            return GraderInitResponse()
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e
    return app

