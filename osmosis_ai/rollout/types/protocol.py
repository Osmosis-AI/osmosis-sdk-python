from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from osmosis_ai.rollout.types.sample import (
    MessageDict,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)


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


class RolloutInitResponse(BaseModel): ...


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
    artifacts: dict[str, Any] | None = None
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
