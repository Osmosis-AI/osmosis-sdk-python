from enum import Enum
from typing import Any

from pydantic import BaseModel

from osmosis_ai.rollout_v2.types.sample import (
    MessageDict,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)


class RolloutInitRequest(BaseModel):
    initial_messages: list[MessageDict]
    rollout_id: str

    chat_completions_url: str
    completion_callback_url: str
    controller_api_key: str | None = None

    label: str | None = None
    grader_callback_url: str | None = None

    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None


class RolloutInitResponse(BaseModel): ...


class RolloutCompleteRequest(BaseModel):
    rollout_id: str
    status: RolloutStatus

    extra_fields: dict[str, Any] | None = None

    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None


class GraderInitRequest(BaseModel):
    rollout_id: str
    samples: dict[str, RolloutSample]
    completion_callback_url: str

    extra_fields: dict[str, Any] | None = None
    controller_api_key: str | None = None


class GraderInitResponse(BaseModel): ...


class GraderStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class GraderCompleteRequest(BaseModel):
    rollout_id: str
    status: GraderStatus
    samples: dict[str, RolloutSample]
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
