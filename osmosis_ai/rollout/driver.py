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


__all__ = ["RolloutDriver", "RolloutOutcome"]
