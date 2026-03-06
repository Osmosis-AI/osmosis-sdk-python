from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

MessageDict = Dict[str, List | str]


class RolloutSample(BaseModel):
    id: str
    messages: List[MessageDict] = Field(default_factory=list)
    label: str | None = None
    reward: float | None = None

    remove_sample: bool = False

    metrics: Dict[str, Any] = Field(default_factory=dict)
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


class RolloutStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class RolloutInitRequest(BaseModel):
    initial_messages: List[MessageDict]
    rollout_id: str

    chat_completions_url: str
    completion_callback_url: str


class RolloutInitResponse(BaseModel):
    ...


class RolloutCompleteRequest(BaseModel):
    rollout_id: str
    status: RolloutStatus

    extra_fields: Optional[Dict[str, Any]] = None

    err_message: str | None = None
    err_category: str | None = None


class MultiTurnMode(str, Enum):
    MULTI_SAMPLE = "multi_sample"
    SINGLE_SAMPLE = "single_sample"
