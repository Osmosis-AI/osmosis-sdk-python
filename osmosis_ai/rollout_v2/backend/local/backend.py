import copy
import logging
import traceback
from typing import Any

from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout_v2.context import (
    AgentWorkflowContext,
    GraderContext,
    RolloutContext,
    get_rollout_context,
)
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    ExecutionRequest,
    ExecutionResult,
    GraderConfig,
    RolloutErrorCategory,
    RolloutStatus,
)
from osmosis_ai.rollout_v2.utils.concurrency import ConcurrencyLimiter
from osmosis_ai.rollout_v2.utils.imports import resolve_object
from osmosis_ai.rollout_v2.utils.messages import map_initial_messages_to_content_blocks

logger = logging.getLogger(__name__)


class LocalBackend(ExecutionBackend):
    def __init__(
        self,
        *,
        workflow: type[AgentWorkflow] | str,
        workflow_config: AgentWorkflowConfig | str | None = None,
        grader: type[Grader] | str | None = None,
        grader_config: GraderConfig | str | None = None,
    ) -> None:
        self.workflow_cls: type[AgentWorkflow] = resolve_object(workflow)
        self.workflow_config: AgentWorkflowConfig | None = (
            resolve_object(workflow_config) if workflow_config else None
        )
        self.grader_cls: type[Grader] | None = (
            resolve_object(grader) if grader else None
        )
        self.grader_config: GraderConfig | None = (
            resolve_object(grader_config) if grader_config else None
        )

        max_concurrent = (
            self.workflow_config.concurrency.max_concurrent
            if self.workflow_config
            else 4
        )
        self.limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "concurrency": self.limiter.snapshot(),
        }

    async def execute(
        self,
        request: ExecutionRequest,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None = None,
    ) -> None:
        async with self.limiter.acquire():
            result = await self.run_workflow(request)
            await on_workflow_complete(result)

            if (
                on_grader_complete
                and self.grader_cls
                and request.label is not None
                and result.status == RolloutStatus.SUCCESS
            ):
                graded = await self.run_grader(request, result)
                await on_grader_complete(graded)

    async def run_workflow(self, request: ExecutionRequest) -> ExecutionResult:
        config = copy.deepcopy(self.workflow_config)
        ctx = AgentWorkflowContext(
            prompt=map_initial_messages_to_content_blocks(request.prompt),
            config=config,
        )

        rollout_ctx = get_rollout_context()
        if rollout_ctx is None:
            rollout_ctx = RolloutContext()

        workflow = self.workflow_cls(config)
        try:
            with rollout_ctx:
                await workflow.run(ctx)
        except Exception as e:
            logger.error(traceback.format_exc())
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=str(e),
                err_category=_categorize_exception(e),
            )

        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            samples=rollout_ctx.get_samples(),
        )

    async def run_grader(
        self, request: ExecutionRequest, result: ExecutionResult
    ) -> ExecutionResult:
        if not self.grader_cls:
            return result

        grader_ctx = GraderContext(label=request.label, samples=result.samples)
        try:
            grader = self.grader_cls(self.grader_config)
            await grader.grade(grader_ctx)
            return ExecutionResult(
                status=RolloutStatus.SUCCESS,
                samples=grader_ctx.get_samples(),
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                samples=result.samples,
                err_message=str(e),
                err_category=_categorize_exception(e),
            )


def _categorize_exception(exc: Exception) -> RolloutErrorCategory:
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    return RolloutErrorCategory.AGENT_ERROR
