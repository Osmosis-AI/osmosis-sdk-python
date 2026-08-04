from enum import StrEnum
from typing import Any

from pydantic import BaseModel, field_validator

from osmosis_ai.rollout.types.sample import (
    MessageDict,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)
from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment


class RolloutInitRequest(BaseModel):
    """Body of the POST to the SDK's /rollout endpoint.

    Routing identity lives in the URLs the caller hands us: the rollout
    id is baked into ``chat_completions_url`` (session-scoped) and into
    both callback URLs as a path segment. ``rollout_id`` is repeated in
    the body for debug logging on the rollout-server side and for
    correlation in user-side dashboards; the SDK does not rely on it for
    routing, so it is optional.
    """

    initial_messages: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None

    rollout_id: str | None = None

    chat_completions_url: str
    controller_api_key: str | None = None
    completion_callback_url: str
    grader_callback_url: str | None = None

    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None

    extra_fields: dict[str, Any] | None = None

    @field_validator("rollout_id")
    @classmethod
    def _validate_rollout_id(cls, value: str | None) -> str | None:
        # ``rollout_id`` is optional on this branch; only validate when given.
        if value is None:
            return None
        return ensure_single_path_segment(value, label="rollout_id")


class RolloutInitResponse(BaseModel): ...


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


class RolloutCompleteRequest(BaseModel):
    """Body of the rollout-complete callback.

    ``rollout_id`` mirrors the id embedded in the callback URL purely for
    debug/log correlation on the controller side. The receiver identifies
    the rollout from the URL path, not this field.
    """

    status: RolloutStatus
    rollout_id: str | None = None

    extra_fields: dict[str, Any] | None = None

    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None


class GraderInitRequest(BaseModel):
    """Body of a POST to a remote grader endpoint.

    A rollout produces a single sample (the agent's conversation), so this
    carries that one sample directly rather than the legacy
    ``dict[str, RolloutSample]``.
    """

    sample: RolloutSample
    rollout_id: str | None = None
    completion_callback_url: str

    extra_fields: dict[str, Any] | None = None
    controller_api_key: str | None = None


class GraderInitResponse(BaseModel): ...


class GraderStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class GraderCompleteRequest(BaseModel):
    """Body of the grader-complete callback.

    Carries the single graded sample (with its ``reward`` populated on
    success) and nothing on failure. ``rollout_id`` mirrors the URL path
    segment for debug/log correlation; the controller resolves the
    rollout from the URL.
    """

    status: GraderStatus
    rollout_id: str | None = None
    sample: RolloutSample | None = None
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
