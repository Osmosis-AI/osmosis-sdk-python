import copy
import logging
import traceback
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.base import ExecutionBackend, ResultCallback
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    RolloutContext,
    get_rollout_context,
)
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    ExecutionRequest,
    ExecutionResult,
    GraderConfig,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.concurrency import ConcurrencyLimiter
from osmosis_ai.rollout.utils.errors import categorize_exception
from osmosis_ai.rollout.utils.file_artifacts import (
    create_rollout_artifacts_dir,
    default_artifact_root,
)
from osmosis_ai.rollout.utils.imports import resolve_object
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward

logger: logging.Logger = logging.getLogger(__name__)


class LocalBackend(ExecutionBackend):
    def __init__(
        self,
        *,
        workflow: type[AgentWorkflow[Any]] | str,
        workflow_config: AgentWorkflowConfig | str | None = None,
        grader: type[Grader] | str | None = None,
        grader_config: GraderConfig | str | None = None,
    ) -> None:
        self.workflow_cls: type[AgentWorkflow[Any]] = resolve_object(workflow)
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
        self.limiter: ConcurrencyLimiter = ConcurrencyLimiter(
            max_concurrent=max_concurrent
        )

        self.artifact_root: Path = default_artifact_root()

    @property
    def max_concurrency(self) -> int:
        return self.limiter.max_concurrent or 0

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

            if not on_grader_complete:
                return

            if (
                self.grader_cls
                and (request.label is not None or request.metadata is not None)
                and result.status == RolloutStatus.SUCCESS
            ):
                graded = await self.run_grader(request, result)
                await on_grader_complete(graded)
            else:
                await on_grader_complete(ExecutionResult(status=RolloutStatus.FAILURE))

    async def run_workflow(self, request: ExecutionRequest) -> ExecutionResult:
        config = copy.deepcopy(self.workflow_config)
        ctx = AgentWorkflowContext(
            prompt=request.prompt,
            config=config,
            metadata=request.metadata,
            artifacts_dir=await create_rollout_artifacts_dir(
                self.artifact_root, request.id
            ),
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
                err_category=categorize_exception(e),
            )

        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=await rollout_ctx.get_sample(),
        )

    async def run_grader(
        self, request: ExecutionRequest, result: ExecutionResult
    ) -> ExecutionResult:
        if not self.grader_cls:
            return result

        grader_ctx = GraderContext(
            label=request.label,
            sample=result.sample,
            metadata=request.metadata,
            artifacts_dir=await create_rollout_artifacts_dir(
                self.artifact_root, request.id
            ),
        )
        try:
            grader = self.grader_cls(self.grader_config)
            await grader.grade(grader_ctx)
            validate_sample_has_reward(grader_ctx.sample)
            return ExecutionResult(
                status=RolloutStatus.SUCCESS,
                sample=grader_ctx.sample,
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=result.sample,
                err_message=str(e),
                err_category=categorize_exception(e),
            )
