from typing import Any, Dict, Optional

from pydantic import BaseModel

from osmosis_ai.rollout_v2.types.rollout import RolloutSample


class GraderInitRequest(BaseModel):
    rollout_id: str
    samples: Dict[str, RolloutSample]
    completion_callback_url: str

    extra_fields: Optional[Dict[str, Any]] = None


class GraderInitResponse(BaseModel):
    ...


class GraderCompleteRequest(BaseModel):
    rollout_id: str
    samples: Dict[str, RolloutSample]
