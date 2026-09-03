import math
from typing import Any, Self

from pydantic import BaseModel, field_validator, model_validator

from osmosis_ai.rollout.types.sample import (
    MessageDict,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

POLLING_LEASE_HEADER = "X-Osmosis-Rollout-Lease"


class RolloutInitRequest(BaseModel):
    initial_messages: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None

    rollout_id: str

    chat_completions_url: str
    llm_api_key: str | None = None
    grade: bool = True

    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None

    extra_fields: dict[str, Any] | None = None

    @field_validator("rollout_id")
    @classmethod
    def _validate_rollout_id(cls, value: str) -> str:
        return ensure_single_path_segment(value, label="rollout_id")

    @field_validator("llm_api_key")
    @classmethod
    def _validate_llm_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            raise ValueError("llm_api_key must be omitted or a non-empty string")
        return value

    @field_validator("agent_timeout_sec", "grader_timeout_sec")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("timeout must be finite; omit it to run unbounded")
        return value


class RolloutInitResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus
    result_wait_timeout_sec: float
    polling_lease_timeout_sec: float

    @model_validator(mode="after")
    def _validate_timeouts(self) -> Self:
        wait = self.result_wait_timeout_sec
        lease = self.polling_lease_timeout_sec
        if not math.isfinite(wait) or wait <= 0:
            raise ValueError("result_wait_timeout_sec must be positive and finite")
        if not math.isfinite(lease) or lease <= wait:
            raise ValueError(
                "polling_lease_timeout_sec must be finite and exceed "
                "result_wait_timeout_sec"
            )
        return self


class RolloutResultResponse(BaseModel):
    """Current or finished result from ``GET /rollout/{rollout_id}/result``.

    In-progress requests contain only their current status. Finished requests
    may also contain the sample and error details.
    """

    rollout_id: str
    status: RolloutStatus
    sample: RolloutSample | None = None
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None


class CancelRolloutsRequest(BaseModel):
    """Body of the POST to /rollout/cancel. Exactly one selector applies:
    explicit ``ids``, an id ``prefix``, or ``all``."""

    ids: list[str] | None = None
    prefix: str | None = None
    all: bool = False


class CancelRolloutsResponse(BaseModel):
    """Disposition per rollout: ``cancelled_queued``, ``cancelled_running``,
    or ``not_found`` for unknown/finished ids (cancellation is idempotent)."""

    dispositions: dict[str, str]
