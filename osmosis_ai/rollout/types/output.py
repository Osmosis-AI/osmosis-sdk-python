"""Return type for AgentWorkflow.run."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

Messages = list[dict[str, Any]]


class AgentWorkflowOutput(BaseModel):
    """What a workflow hands back: one message history plus optional measurements.

    One rollout produces one sample, so ``messages`` is a single message
    history (``None`` when the workflow produced none). ``info`` is reserved
    for workflow-specific metadata; current backends do not pass it to
    graders.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    messages: Messages | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


def coerce_output(value: Any) -> AgentWorkflowOutput | None:
    """Normalize and validate a run() return value.

    ``None`` means "use the fallback source".
    """
    if value is None:
        return None
    if isinstance(value, AgentWorkflowOutput):
        return AgentWorkflowOutput.model_validate(value)
    if isinstance(value, list):
        return AgentWorkflowOutput(messages=value)
    raise TypeError(
        "run() must return AgentWorkflowOutput, a message list, or None; "
        f"got {type(value).__name__}"
    )
