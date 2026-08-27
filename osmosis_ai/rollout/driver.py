"""Eval-facing types for a single rollout execution.

``HttpRolloutDriver`` is to eval what the trainer is to the rollout server:
it provides data + LLM endpoint and consumes trace + reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osmosis_ai.rollout.types import MessageDict, RolloutSample, RolloutStatus


@dataclass
class RolloutOutcome:
    """Result of a single rollout execution.

    Single-sample by design: one rollout = one agent run = one reward.
    ``sample`` carries the conversation and reward (when grading succeeded),
    ``rollout_id`` identifies the rollout in logs/cache rows.
    """

    status: RolloutStatus
    sample: RolloutSample | None = None
    error: str | None = None
    rollout_id: str | None = None


@dataclass
class RolloutRunRequest:
    """Inputs for one ``HttpRolloutDriver.run`` call."""

    messages: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None
    rollout_id: str = ""
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None
    extra_fields: dict[str, Any] | None = None


__all__ = ["RolloutOutcome", "RolloutRunRequest"]
