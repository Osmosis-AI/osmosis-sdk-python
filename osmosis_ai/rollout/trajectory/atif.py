"""Internal ATIF v1.7 models used by trajectory persistence.

These models were initially adapted from Harbor 0.20.0:

- ``harbor/models/trajectories/`` for the ATIF models
- ``harbor/utils/trajectory_utils.py`` for JSON formatting

They intentionally live in the SDK because trajectory persistence is shared by
all execution backends. Importing Harbor's models here would make the generic
rollout server and ``LocalBackend`` require the optional ``harbor`` extra.
This module is not re-exported as supported SDK API; external consumers that
need general-purpose ATIF models should use Harbor's implementation directly.

When updating ATIF support, compare this module with both the upstream ATIF RFC
and those Harbor paths. Preserve SDK-specific hardening such as rejecting
non-finite floats, which prevents invalid JSON documents.

Every document the SDK writes goes through these models — including the one the
in-container runner leaves for the Harbor backend to collect. The one place that
reads back Harbor's own models is ``HarborBackend.load_native_trajectory``,
which parses documents that Harbor's native agents authored; keep the two in
step when bumping the pinned Harbor line.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__: list[str] = []

ATIFSchemaVersion = Literal[
    "ATIF-v1.0",
    "ATIF-v1.1",
    "ATIF-v1.2",
    "ATIF-v1.3",
    "ATIF-v1.4",
    "ATIF-v1.5",
    "ATIF-v1.6",
    "ATIF-v1.7",
]


class _ATIFModel(BaseModel):
    """Base configuration shared by all ATIF document objects."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class ImageSource(_ATIFModel):
    """A file or URL containing an image used in multimodal content."""

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class ContentPart(_ATIFModel):
    """One text or image part in a multimodal message."""

    type: Literal["text", "image"]
    text: str | None = None
    source: ImageSource | None = None

    @model_validator(mode="after")
    def _validate_content_type(self) -> Self:
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
        else:
            if self.source is None:
                raise ValueError("'source' field is required when type='image'")
            if self.text is not None:
                raise ValueError("'text' field is not allowed when type='image'")
        return self


class Agent(_ATIFModel):
    """Agent configuration recorded at the trajectory root."""

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class Metrics(_ATIFModel):
    """Operational data for one LLM inference."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class FinalMetrics(_ATIFModel):
    """Aggregate metrics for a complete trajectory."""

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class ToolCall(_ATIFModel):
    """A structured tool invocation made during an agent step."""

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class SubagentTrajectoryRef(_ATIFModel):
    """A resolvable reference to a delegated agent's trajectory."""

    trajectory_id: str | None = None
    session_id: str | None = None
    trajectory_path: str | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_is_resolvable(self) -> Self:
        if self.trajectory_id is None and self.trajectory_path is None:
            raise ValueError(
                "SubagentTrajectoryRef must be resolvable: set either "
                "`trajectory_id` (for embedded references) or "
                "`trajectory_path` (for external-file references). "
                "`session_id` alone is not a resolution key -- it is "
                "run-scoped and may collide across siblings."
            )
        return self


class ObservationResult(_ATIFModel):
    """One tool, environment, or delegated-agent result."""

    source_call_id: str | None = None
    content: str | list[ContentPart] | None = None
    subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = None
    extra: dict[str, Any] | None = None


class Observation(_ATIFModel):
    """Environment feedback attached to a trajectory step."""

    results: list[ObservationResult]


class Step(_ATIFModel):
    """One sequential turn in an ATIF trajectory."""

    step_id: int = Field(ge=1)
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str | list[ContentPart]
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid ISO 8601 timestamp: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _validate_agent_only_fields(self) -> Self:
        if self.source == "agent":
            return self
        for field_name in (
            "model_name",
            "reasoning_effort",
            "reasoning_content",
            "tool_calls",
            "metrics",
        ):
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"Field '{field_name}' is only applicable when source is "
                    f"'agent', but source is '{self.source}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_zero_llm_call_fields(self) -> Self:
        if self.source == "agent" and self.llm_call_count == 0:
            for field_name in ("metrics", "reasoning_content"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"Field '{field_name}' must be absent when llm_call_count "
                        "is 0 (deterministic dispatch on a 'source: agent' step)"
                    )
        return self


class Trajectory(_ATIFModel):
    """A complete Agent Trajectory Interchange Format document."""

    schema_version: ATIFSchemaVersion = "ATIF-v1.7"
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: Agent
    steps: list[Step] = Field(min_length=1)
    notes: str | None = None
    final_metrics: FinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list[Trajectory] | None = None

    def to_json_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible dictionary for persistence."""
        return self.model_dump(exclude_none=exclude_none, mode="json")

    @model_validator(mode="after")
    def _validate_step_ids(self) -> Self:
        for index, step in enumerate(self.steps):
            expected_step_id = index + 1
            if step.step_id != expected_step_id:
                raise ValueError(
                    f"steps[{index}].step_id: expected {expected_step_id} "
                    f"(sequential from 1), got {step.step_id}"
                )
        return self

    @model_validator(mode="after")
    def _validate_embedded_subagent_trajectory_ids(self) -> Self:
        if not self.subagent_trajectories:
            return self
        seen: set[str] = set()
        for index, subagent in enumerate(self.subagent_trajectories):
            trajectory_id = subagent.trajectory_id
            if trajectory_id is None:
                raise ValueError(
                    f"subagent_trajectories[{index}].trajectory_id is required "
                    "for embedded subagents "
                    f"(agent.name={subagent.agent.name!r}, "
                    f"session_id={subagent.session_id!r})"
                )
            if trajectory_id in seen:
                raise ValueError(
                    f"subagent_trajectories[{index}].trajectory_id "
                    f"{trajectory_id!r} is not unique within subagent_trajectories"
                )
            seen.add(trajectory_id)
        return self

    @model_validator(mode="after")
    def _validate_tool_call_references(self) -> Self:
        for step in self.steps:
            if step.observation is None:
                continue
            tool_call_ids = (
                {call.tool_call_id for call in step.tool_calls}
                if step.tool_calls
                else set()
            )
            for result in step.observation.results:
                source_call_id = result.source_call_id
                if source_call_id is not None and source_call_id not in tool_call_ids:
                    raise ValueError(
                        "Observation result references source_call_id "
                        f"'{source_call_id}' which is not found in step "
                        f"{step.step_id}'s tool_calls"
                    )
        return self

    def has_multimodal_content(self) -> bool:
        """Return whether any step or observation contains an image."""
        for step in self.steps:
            if isinstance(step.message, list) and any(
                part.type == "image" for part in step.message
            ):
                return True
            if step.observation is not None:
                for result in step.observation.results:
                    if isinstance(result.content, list) and any(
                        part.type == "image" for part in result.content
                    ):
                        return True
        return False


_NUMERIC_ARRAY_PATTERN = re.compile(
    r"\[\s*\n\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?:\s*,\s*\n\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)*\s*\n\s*\]",
    flags=re.MULTILINE,
)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def format_trajectory_json(data: dict[str, Any]) -> str:
    """Pretty-print a trajectory while keeping numeric arrays on one line."""

    def compact_numeric_array(match: re.Match[str]) -> str:
        return "[" + ", ".join(_NUMBER_PATTERN.findall(match.group(0))) + "]"

    return _NUMERIC_ARRAY_PATTERN.sub(
        compact_numeric_array,
        json.dumps(data, indent=2, allow_nan=False),
    )
