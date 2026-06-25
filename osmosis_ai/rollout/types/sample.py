from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

MessageDict = dict[str, Any]
SampleMessage = Mapping[str, Any]


class RolloutSample(BaseModel):
    """The conversation + grading artefacts produced by one rollout.

    A rollout produces exactly one sample (one agent run, one reward). There
    used to be a per-sample id so the wire protocol could carry a
    ``dict[str, RolloutSample]``; with the URL-routed single-sample wire
    protocol the id is gone and callers identify rollouts via the URL paths
    they hand the SDK.
    """

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


class ExecutionRequest(BaseModel):
    id: str
    prompt: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None


class ExecutionResult(BaseModel):
    status: RolloutStatus
    sample: RolloutSample | None = None
    artifacts: dict[str, Any] | None = None
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
