from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

MessageDict = dict[str, Any]
SampleMessage = Mapping[str, Any]


class RolloutSample(BaseModel):
    id: str
    messages: Sequence[SampleMessage] = Field(default_factory=list)
    label: str | None = None
    reward: float | None = None

    remove_sample: bool = False

    metrics: dict[str, Any] = Field(default_factory=dict)
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class RolloutStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class RolloutErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    HTTP_ERROR = "http_error"
    AGENT_ERROR = "agent_error"


class MultiTurnMode(StrEnum):
    MULTI_SAMPLE = "multi_sample"
    SINGLE_SAMPLE = "single_sample"


class ExecutionRequest(BaseModel):
    id: str
    prompt: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        # ``id`` is joined onto host paths by every backend.
        return ensure_single_path_segment(value, label="rollout_id")


class ExecutionResult(BaseModel):
    status: RolloutStatus
    samples: dict[str, RolloutSample] = Field(default_factory=dict)
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
