import asyncio
import copy
import logging
import time
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
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.types.output import coerce_output
from osmosis_ai.rollout.utils.concurrency import ConcurrencyLimiter
from osmosis_ai.rollout.utils.errors import categorize_exception
from osmosis_ai.rollout.utils.file_artifacts import (
    create_rollout_artifacts_dir,
    default_artifact_root,
)
from osmosis_ai.rollout.utils.imports import resolve_object
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward

logger: logging.Logger = logging.getLogger(__name__)


def _deadline_message(phase: str, limit: float | None, queued_sec: float) -> str:
    message = f"{phase} exceeded its {limit}s deadline"
    if queued_sec >= 0.1:
        message += f" ({queued_sec:.1f}s of it spent queued)"
    return message


def _failure_message(
    exc: Exception,
    deadline: asyncio.Timeout,
    phase: str,
    limit: float | None,
    queued_sec: float = 0.0,
) -> str:
    """Name the deadline that fired, rather than ``str(TimeoutError())``.

    User code may raise its own ``TimeoutError`` (an HTTP client giving up, for
    instance), which carries a useful message and is not our deadline —
    ``expired()`` is what distinguishes the two. The wire category is
    ``TIMEOUT`` either way.
    """
    if deadline.expired():
        return _deadline_message(phase, limit, queued_sec)
    return str(exc)


def _expired_after_return(
    deadline: asyncio.Timeout, started: float, budget: float | None
) -> bool:
    """User code outran its deadline but returned a result anyway.

    ``expired()`` catches a swallowed ``CancelledError``; the wall-clock
    comparison catches sync code that blocked the event loop past the deadline
    so the timeout callback never got to run. Synchronous Python cannot be
    preempted safely — but success after the controller stopped waiting would
    be a lie, so once control returns the result is a timeout.
    """
    if budget is None:
        return False
    return deadline.expired() or time.monotonic() - started > budget


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
        # The controller's clock started at submission, not at slot admission;
        # time spent queued here has to come out of the workflow's budget.
        enqueued = time.monotonic()
        async with self.limiter.acquire():
            queued_sec = time.monotonic() - enqueued
            result = await self.run_workflow(request, queued_sec=queued_sec)
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

    async def run_workflow(
        self, request: ExecutionRequest, *, queued_sec: float = 0.0
    ) -> ExecutionResult:
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
        # The controller expires its own session at this deadline; without it
        # here the workflow keeps running — and keeps its concurrency slot —
        # long after anyone is waiting for the answer. The controller's clock
        # covers queue time too, so only what the queue left over is available.
        # ``None`` means the controller sent no deadline, so run unbounded.
        #
        # This stops a cooperatively-cancellable workflow. One that swallows
        # ``CancelledError``, or that blocks the event loop in sync code, still
        # runs to completion; the rollout is reported as a timeout either way.
        budget = (
            max(request.agent_timeout_sec - queued_sec, 0.0)
            if request.agent_timeout_sec is not None
            else None
        )
        started = time.monotonic()
        deadline = asyncio.timeout(budget)
        try:
            async with deadline:
                with rollout_ctx:
                    output = coerce_output(await workflow.run(ctx))
                    if output is None:
                        sample = await rollout_ctx.get_sample()
                    else:
                        sample = (
                            RolloutSample(
                                messages=output.messages,
                                label=request.label,
                                metrics=dict(output.metrics),
                            )
                            if output.messages is not None
                            else None
                        )
        except Exception as e:
            logger.error(traceback.format_exc())
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=_failure_message(
                    e, deadline, "workflow", request.agent_timeout_sec, queued_sec
                ),
                err_category=categorize_exception(e),
            )

        if _expired_after_return(deadline, started, budget):
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message=_deadline_message(
                    "workflow", request.agent_timeout_sec, queued_sec
                ),
                err_category=RolloutErrorCategory.TIMEOUT,
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=sample,
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
        # Graded independently of the agent deadline: a workflow that finished
        # just inside its budget still gets its full grading window, and a hung
        # grader cannot hold the slot either.
        budget = request.grader_timeout_sec
        started = time.monotonic()
        deadline = asyncio.timeout(budget)
        try:
            async with deadline:
                grader = self.grader_cls(self.grader_config)
                await grader.grade(grader_ctx)
                validate_sample_has_reward(grader_ctx.sample)
        except Exception as e:
            logger.error(traceback.format_exc())
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=result.sample,
                err_message=_failure_message(e, deadline, "grader", budget),
                err_category=categorize_exception(e),
            )

        if _expired_after_return(deadline, started, budget):
            return ExecutionResult(
                status=RolloutStatus.FAILURE,
                sample=result.sample,
                err_message=_deadline_message("grader", budget, 0.0),
                err_category=RolloutErrorCategory.TIMEOUT,
            )
        return ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=grader_ctx.sample,
        )
