"""Trial naming, in-flight state, and Harbor hook-event readers.

Shared by both Harbor backends, so neither depends on the other.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from harbor.models.trial.result import ExceptionInfo
from harbor.trial.hooks import TrialHookEvent

from osmosis_ai.rollout.backend.base import ResultCallback
from osmosis_ai.rollout.types import ExecutionResult

logger: logging.Logger = logging.getLogger(__name__)

TRIAL_NAME_PREFIX = "trial-"


class PendingTrial:
    def __init__(
        self,
        on_workflow_complete: ResultCallback,
        on_grader_complete: ResultCallback | None,
    ):
        self.on_workflow_complete = on_workflow_complete
        self.on_grader_complete = on_grader_complete
        # The workflow outcome is produced at verification start and cached so
        # a failed delivery is retried byte-identical at trial end; *_called
        # means "delivered". The grader outcome is produced only at trial end,
        # so it needs no cache -- just double-send protection.
        self.workflow_complete_called = False
        self.workflow_result: ExecutionResult | None = None
        self.grader_complete_called = False
        # Lets execute() tell a requested cancellation from an external one.
        self.cancel_requested = False
        # Request label, the fallback when the sample carries none.
        self.label: str | None = None
        self.preserve_trial = False
        self.started = False
        self.grading = False
        self.api_key: str | None = None
        self.task: asyncio.Task[Any] | None = None
        self.done: asyncio.Future[None] = asyncio.get_event_loop().create_future()


def parse_rollout_id(event: TrialHookEvent) -> str:
    return event.config.trial_name.removeprefix(TRIAL_NAME_PREFIX)


def get_agent_metadata(event: TrialHookEvent) -> dict[str, Any] | None:
    if event.result and event.result.agent_result:
        return event.result.agent_result.metadata
    return None


def log_trial_exception(rollout_id: str, err: ExceptionInfo, *, phase: str) -> None:
    """Log a harbor trial exception in full.

    ``ExecutionResult`` carries only the message, and Harbor's own copy lives in
    the trial directory, which cleanup removes.
    """
    logger.error(
        "Harbor trial %s failed %s [%s]: %s\n%s",
        rollout_id,
        phase,
        err.exception_type,
        err.exception_message,
        err.exception_traceback,
    )
