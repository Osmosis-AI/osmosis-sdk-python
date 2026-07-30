import copy
import logging
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

MessageDict = dict[str, Any]
SampleMessage = Mapping[str, Any]


class RolloutSample(BaseModel):
    """The conversation + grading artefacts produced by one rollout.

    A rollout produces exactly one sample (one agent run, one reward). There
    used to be a per-sample id so the wire protocol could carry a
    ``dict[str, RolloutSample]``; with the URL-routed single-sample wire
    protocol the id is gone and callers identify rollouts via the URL paths
    they hand the SDK.
    """

    messages: Sequence[SampleMessage] = Field(default_factory=list)
    trajectory_messages: Sequence[SampleMessage] | None = None
    label: str | None = None
    reward: float | None = None

    remove_sample: bool = False

    metrics: dict[str, Any] = Field(default_factory=dict)
    extra_fields: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_trajectory_messages(self) -> Self:
        # Explicit None disables trajectory persistence.
        if "trajectory_messages" not in self.model_fields_set:
            try:
                self.trajectory_messages = copy.deepcopy(list(self.messages))
            except Exception:
                logger.warning(
                    "Failed to snapshot messages for trajectory persistence",
                    exc_info=True,
                )
                self.trajectory_messages = None
        return self


class RolloutStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class RolloutErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    HTTP_ERROR = "http_error"
    AGENT_ERROR = "agent_error"


class ExecutionRequest(BaseModel):
    id: str
    prompt: list[MessageDict]
    label: str | None = None
    metadata: dict[str, Any] | None = None
    agent_timeout_sec: float | None = None
    grader_timeout_sec: float | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        # ``id`` is joined onto host paths by every backend.
        return ensure_single_path_segment(value, label="rollout_id")


class ExecutionResult(BaseModel):
    status: RolloutStatus
    sample: RolloutSample | None = None
    # Backend-produced callback diagnostics. Unlike request/sample extra fields,
    # these describe execution of the rollout itself and are archived separately.
    extra_fields: dict[str, Any] | None = None
    # A backend-native ATIF document that is already richer than the SDK's
    # chat-message normalization. It is process-local execution state: callback
    # payloads are built explicitly by the server, and model serialization must
    # never expose the document (which can be large and may contain agent config).
    trajectory_document: dict[str, Any] | None = Field(default=None, exclude=True)
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
