"""RolloutDriver — eval-facing protocol for rollout execution.

RolloutDriver is to eval what the trainer is to the rollout server:
it provides data + LLM endpoint and consumes trace + reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.rollout.types import RolloutSample, RolloutStatus


@dataclass
class RolloutOutcome:
    """Result of a single rollout execution.

    Single-sample by design: one rollout = one agent run = one reward.
    ``sample`` carries the conversation and reward (when grading succeeded),
    ``rollout_id`` identifies the rollout in logs/cache rows, and the rest
    are run-level metrics + diagnostics.
    """

    status: RolloutStatus
    sample: RolloutSample | None = None
    error: str | None = None
    duration_ms: float = 0.0
    tokens: int = 0
    systemic_error: str | None = None
    rollout_id: str | None = None
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


__all__ = ["RolloutDriver", "RolloutOutcome"]
