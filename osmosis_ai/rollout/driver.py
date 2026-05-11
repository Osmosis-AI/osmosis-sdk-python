"""RolloutDriver — eval-facing protocol for rollout execution.

RolloutDriver is to eval what the trainer is to the rollout server:
it provides data + LLM endpoint and consumes trace + reward.

InProcessDriver mirrors app.py:_handle_rollout() but collects results
via in-process callbacks instead of HTTP.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from osmosis_ai.rollout.types import RolloutSample, RolloutStatus

if TYPE_CHECKING:
    from osmosis_ai.rollout.backend.base import ExecutionBackend


@dataclass
class RolloutOutcome:
    """Result of a single rollout execution.

    This is the eval-facing contract. Eval doesn't need to know
    whether this came from in-process execution or an HTTP call.
    """

    status: RolloutStatus
    samples: dict[str, RolloutSample] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    tokens: int = 0
    systemic_error: str | None = None
    rollout_id: str | None = None
    controller_created_sample_ids: list[str] = field(default_factory=list)
    completion_counts: dict[str, int] = field(default_factory=dict)
    full_callback_sample_ids: list[str] = field(default_factory=list)
    scored_sample_ids: list[str] = field(default_factory=list)
    skipped_sample_ids: list[str] = field(default_factory=list)
    callback_diagnostics: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False


class RolloutDriver(ABC):
    """Drives a rollout execution."""

    @property
    def max_concurrency(self) -> int:
        """Max concurrent executions. 0 = no limit."""
        return 0

    @abstractmethod
    async def run(
        self,
        messages: list[dict[str, Any]],
        label: str | None = None,
        rollout_id: str = "",
    ) -> RolloutOutcome:
        raise NotImplementedError


class InProcessDriver(RolloutDriver):
    """In-process rollout execution via ExecutionBackend.

    Mirrors app.py:_handle_rollout() — sets up RolloutContext,
    constructs ExecutionRequest, delegates to backend.execute(),
    and adapts callback results to RolloutOutcome.
    """

    def __init__(self, *, backend: ExecutionBackend, proxy: Any) -> None:
        self.backend = backend
        self.proxy = proxy

    @property
    def max_concurrency(self) -> int:
        return self.backend.max_concurrency

    async def run(
        self,
        messages: list[dict[str, Any]],
        label: str | None = None,
        rollout_id: str = "",
    ) -> RolloutOutcome:
        from osmosis_ai.rollout.context import RolloutContext, rollout_contextvar
        from osmosis_ai.rollout.types import ExecutionRequest, ExecutionResult

        start = time.monotonic()
        effective_rollout_id = rollout_id or f"in-process-{uuid.uuid4().hex}"

        rollout_ctx = RolloutContext(
            chat_completions_url=self.proxy.url,
            api_key=self.proxy.api_key,
            rollout_id=effective_rollout_id,
        )

        request = ExecutionRequest(
            id=effective_rollout_id,
            prompt=messages,
            label=label,
        )

        workflow_result: ExecutionResult | None = None
        grader_result: ExecutionResult | None = None

        async def on_workflow_complete(result: ExecutionResult) -> None:
            nonlocal workflow_result
            workflow_result = result

        async def on_grader_complete(result: ExecutionResult) -> None:
            nonlocal grader_result
            grader_result = result

        token = rollout_contextvar.set(rollout_ctx)
        try:
            await self.backend.execute(
                request,
                on_workflow_complete=on_workflow_complete,
                on_grader_complete=on_grader_complete,
            )
        except Exception as e:
            systemic = self.proxy.collect_systemic_error(effective_rollout_id)
            return RolloutOutcome(
                status=RolloutStatus.FAILURE,
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
                systemic_error=systemic,
                rollout_id=effective_rollout_id,
            )
        finally:
            rollout_contextvar.reset(token)

        final = grader_result if grader_result is not None else workflow_result
        if final is None:
            return RolloutOutcome(
                status=RolloutStatus.FAILURE,
                error="No result from backend",
                duration_ms=(time.monotonic() - start) * 1000,
                rollout_id=effective_rollout_id,
            )

        tokens = self.proxy.collect_tokens(effective_rollout_id)
        systemic = self.proxy.collect_systemic_error(effective_rollout_id)
        samples = cast(dict[str, RolloutSample], final.samples)
        sample_ids = sorted(samples)
        scored_sample_ids: list[str] = []
        skipped_sample_ids: list[str] = []
        for sample_id in samples:
            rollout_sample = samples[sample_id]
            if rollout_sample.remove_sample:
                skipped_sample_ids.append(sample_id)
            else:
                scored_sample_ids.append(sample_id)
        scored_sample_ids.sort()
        skipped_sample_ids.sort()

        return RolloutOutcome(
            status=final.status,
            samples=samples,
            error=final.err_message,
            duration_ms=(time.monotonic() - start) * 1000,
            tokens=tokens,
            systemic_error=systemic,
            rollout_id=effective_rollout_id,
            full_callback_sample_ids=sample_ids,
            scored_sample_ids=scored_sample_ids,
            skipped_sample_ids=skipped_sample_ids,
            skipped=final.status == RolloutStatus.SUCCESS
            and bool(samples)
            and not scored_sample_ids,
        )


__all__ = ["InProcessDriver", "RolloutDriver", "RolloutOutcome"]
