"""Return type for AgentWorkflow.run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

Messages = list[dict[str, Any]]


class AgentWorkflowOutput(BaseModel):
    """What a workflow hands back: message histories plus optional measurements.

    ``samples`` maps a name to one agent's message history (multi-agent
    workflows return several). ``info`` carries workflow-produced context for
    the grader; the rollout request's ``metadata`` is a separate, input-side
    field.
    """

    samples: dict[str, Messages] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)

    def primary_messages(self) -> Messages | None:
        if not self.samples:
            return None
        if "default" in self.samples:
            return self.samples["default"]
        return next(iter(self.samples.values()))


def coerce_output(value: Any) -> AgentWorkflowOutput | None:
    """Normalize a run() return value; None means "use the fallback source"."""
    if value is None:
        return None
    if isinstance(value, AgentWorkflowOutput):
        return value
    if isinstance(value, list):
        return AgentWorkflowOutput(samples={"default": value})
    raise TypeError(
        "run() must return AgentWorkflowOutput, a message list, or None; "
        f"got {type(value).__name__}"
    )
