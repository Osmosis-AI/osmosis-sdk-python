from enum import Enum
from typing import Any

from pydantic import BaseModel

from osmosis_ai.rollout_v2.types.rollout import RolloutErrorCategory, RolloutSample


class GraderInitRequest(BaseModel):
    rollout_id: str
    samples: dict[str, RolloutSample]
    completion_callback_url: str

    extra_fields: dict[str, Any] | None = None


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
