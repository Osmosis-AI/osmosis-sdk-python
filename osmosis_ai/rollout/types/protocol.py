import math
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

    ``controller_api_key`` authenticates callbacks back to the controller
    listener. ``llm_api_key`` authenticates the agent to
    ``chat_completions_url`` — locally the LiteLLM bridge's per-run bearer,
    a genuinely different secret from the callback bearer. When
    ``llm_api_key`` is omitted (``None``), the server falls back to
    ``controller_api_key`` so existing single-secret callers keep working.
    An explicit empty string is rejected; only ``None`` triggers fallback.
    """

    initial_messages: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None

    rollout_id: str | None = None

    chat_completions_url: str
    controller_api_key: str | None = None
    llm_api_key: str | None = None
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

    @field_validator("llm_api_key")
    @classmethod
    def _validate_llm_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            raise ValueError(
                "llm_api_key must be omitted or a non-empty string; "
                "empty string is not a legacy fallback"
            )
        return value

    @field_validator("agent_timeout_sec", "grader_timeout_sec")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        # Rejecting at admission turns a misconfigured deadline into a 422 the
        # controller sees immediately, not an "Internal server error" callback
        # minutes later. Mirrors ExecutionRequest's rule.
        if value is not None and not math.isfinite(value):
            raise ValueError("timeout must be finite; omit it to run unbounded")
        return value


class RolloutInitResponse(BaseModel): ...


class RolloutStatusResponse(BaseModel):
    """GET /rollout/{rollout_id}/status.

    QUEUED/RUNNING/GRADING while in flight; SUCCESS/FAILURE/CANCELLED for a
    retention window after the rollout ends; UNKNOWN for ids never seen or
    whose record aged out.
    """

    rollout_id: str
    status: RolloutStatus
    reward: float | None = None
    err_message: str | None = None


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
