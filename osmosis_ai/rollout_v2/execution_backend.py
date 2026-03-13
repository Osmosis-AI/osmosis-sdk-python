import copy
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any

from fastapi import HTTPException

from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.context import (
    AgentWorkflowContext,
    ControllerAuth,
    GraderContext,
    RolloutContext,
)
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.server_state import ConcurrencyLimiter
from osmosis_ai.rollout_v2.types import (
    AgentWorkflowConfig,
    GraderCompleteRequest,
    GraderConfig,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutErrorCategory,
    RolloutInitRequest,
    RolloutStatus,
)
from osmosis_ai.rollout_v2.utils.http_utils import post_json_with_retry
from osmosis_ai.rollout_v2.utils.misc import map_initial_messages_to_content_blocks

logger = logging.getLogger(__name__)


def _categorize_exception(exc: Exception) -> RolloutErrorCategory:
    if isinstance(exc, TimeoutError):
        return RolloutErrorCategory.TIMEOUT
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return RolloutErrorCategory.VALIDATION_ERROR
    if isinstance(exc, HTTPException):
        return RolloutErrorCategory.HTTP_ERROR
    return RolloutErrorCategory.AGENT_ERROR


class ExecutionBackend(ABC):
    @abstractmethod
    async def execute_rollout(self, request: RolloutInitRequest) -> None:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}


class LocalBackend(ExecutionBackend):
    def __init__(
        self,
        *,
        agent_workflow_cls: type[AgentWorkflow],
        agent_workflow_config: AgentWorkflowConfig,
        grader_cls: type[Grader] | None = None,
        grader_config: GraderConfig | None = None,
    ) -> None:
        self._agent_workflow_cls = agent_workflow_cls
        self._agent_workflow_config = agent_workflow_config
        self._grader_cls = grader_cls
        self._grader_config = grader_config
        self._limiter = ConcurrencyLimiter(
            max_concurrent=agent_workflow_config.concurrency.max_concurrent
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "concurrency": self._limiter.snapshot(),
        }

    async def execute_rollout(self, request: RolloutInitRequest) -> None:
        async with self._limiter.acquire():
            auth_headers = ControllerAuth(
                api_key=request.controller_api_key
            ).as_bearer_headers()

            # 1. Run agent workflow
            config = copy.deepcopy(self._agent_workflow_config)
            rollout_ctx = RolloutContext(
                rollout_id=request.rollout_id,
                chat_completions_url=request.chat_completions_url,
                api_key=request.controller_api_key,
            )
            workflow = self._agent_workflow_cls(config)
            agent_workflow_ctx = AgentWorkflowContext(
                prompt=map_initial_messages_to_content_blocks(request.initial_messages),
                config=config,
            )

            rollout_status = RolloutStatus.SUCCESS
            rollout_err_message = None
            rollout_err_category = None

            try:
                with rollout_ctx:
                    await workflow.run(agent_workflow_ctx)
            except Exception as e:
                logger.error(traceback.format_exc())
                rollout_status = RolloutStatus.FAILURE
                rollout_err_message = str(e)
                rollout_err_category = _categorize_exception(e)

            # 2. POST rollout completion callback
            await post_json_with_retry(
                url=request.completion_callback_url,
                payload=RolloutCompleteRequest(
                    rollout_id=request.rollout_id,
                    status=rollout_status,
                    err_message=rollout_err_message,
                    err_category=rollout_err_category,
                ).model_dump(),
                headers=auth_headers,
            )

            # 3. Run grader + POST grader callback
            if (
                self._grader_cls
                and self._grader_config
                and request.label is not None
                and request.grader_callback_url
                and rollout_status == RolloutStatus.SUCCESS
            ):
                samples = rollout_ctx.get_samples()
                grader_ctx = GraderContext(label=request.label, samples=samples)

                grader_status = GraderStatus.SUCCESS
                grader_err_message = None
                grader_err_category = None

                try:
                    grader = self._grader_cls(self._grader_config)
                    await grader.grade(grader_ctx)

                    graded_samples = grader_ctx.get_samples()
                    assert all(s.reward is not None for s in graded_samples.values()), (
                        "All samples must have a reward after grading"
                    )
                except Exception as e:
                    logger.error(traceback.format_exc())
                    grader_status = GraderStatus.FAILURE
                    grader_err_message = str(e)
                    grader_err_category = _categorize_exception(e)

                await post_json_with_retry(
                    url=request.grader_callback_url,
                    payload=GraderCompleteRequest(
                        rollout_id=request.rollout_id,
                        status=grader_status,
                        samples=grader_ctx.get_samples(),
                        err_message=grader_err_message,
                        err_category=grader_err_category,
                    ).model_dump(),
                    headers=auth_headers,
                )
