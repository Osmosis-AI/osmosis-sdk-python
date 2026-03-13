from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

MessageDict = dict[str, list | str]


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


class RolloutInitRequest(BaseModel):
    initial_messages: list[MessageDict]
    rollout_id: str

    chat_completions_url: str
    completion_callback_url: str
    controller_api_key: str | None = None

    label: str | None = None
    grader_callback_url: str | None = None


class RolloutInitResponse(BaseModel): ...


class RolloutCompleteRequest(BaseModel):
    rollout_id: str
    status: RolloutStatus

    extra_fields: dict[str, Any] | None = None

    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None


class MultiTurnMode(str, Enum):
    MULTI_SAMPLE = "multi_sample"
    SINGLE_SAMPLE = "single_sample"
