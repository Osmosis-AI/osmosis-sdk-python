from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

MessageDict = dict[str, Any]


class RolloutSample(BaseModel):
    id: str
    messages: list[MessageDict] = Field(default_factory=list)
    label: str | None = None
    reward: float | None = None

    remove_sample: bool = False

    metrics: dict[str, Any] = Field(default_factory=dict)
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class RolloutStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class RolloutErrorCategory(str, Enum):
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    HTTP_ERROR = "http_error"
    AGENT_ERROR = "agent_error"


class MultiTurnMode(str, Enum):
    MULTI_SAMPLE = "multi_sample"
    SINGLE_SAMPLE = "single_sample"


class ExecutionRequest(BaseModel):
    id: str
    prompt: list[MessageDict]
    label: str | None = None
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None


class ExecutionResult(BaseModel):
    status: RolloutStatus
    samples: dict[str, RolloutSample] = Field(default_factory=dict)
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
