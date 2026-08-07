"""Return type for AgentWorkflow.run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

Messages = list[dict[str, Any]]


class AgentWorkflowOutput(BaseModel):
    """What a workflow hands back: one message history plus optional measurements.

    ``samples`` contains at most one named message history because one rollout
    produces one sample. ``info`` is reserved for workflow-specific metadata;
    current backends do not pass it to graders.
    """

    samples: dict[str, Messages] = Field(default_factory=dict, max_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)

    def primary_messages(self) -> Messages | None:
        if not self.samples:
            return None
        if "default" in self.samples:
            return self.samples["default"]
        return next(iter(self.samples.values()))


def coerce_output(value: Any) -> AgentWorkflowOutput | None:
    """Normalize and validate a run() return value.

    ``None`` means "use the fallback source".
    """
    if value is None:
        return None
    if isinstance(value, AgentWorkflowOutput):
        if len(value.samples) > 1:
            raise ValueError(
                f"run() returned {len(value.samples)} named samples "
                f"({sorted(value.samples)}); a rollout carries exactly one "
                "sample — return a single message list or one samples entry"
            )
        return value
    if isinstance(value, list):
        return AgentWorkflowOutput(samples={"default": value})
    raise TypeError(
        "run() must return AgentWorkflowOutput, a message list, or None; "
        f"got {type(value).__name__}"
    )
