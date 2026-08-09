import copy
import logging
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

MessageDict = dict[str, Any]
SampleMessage = Mapping[str, Any]


def _is_json_safe(value: Any) -> bool:
    """NaN and infinity have no JSON representation.

    ``json.dumps`` emits the non-standard ``NaN``/``Infinity`` literals and
    HTTPX's encoder raises outright, so one non-finite telemetry value is
    enough to cost a callback its whole payload.
    """
    return not isinstance(value, float) or math.isfinite(value)


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return any(not _is_json_safe(v) or _has_non_finite(v) for v in value.values())
    # Tuples encode as JSON arrays too, so they carry the same hazard.
    if isinstance(value, (list, tuple)):
        return any(not _is_json_safe(v) or _has_non_finite(v) for v in value)
    return False


def _drop_non_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_non_finite(v) for k, v in value.items() if _is_json_safe(v)}
    if isinstance(value, (list, tuple)):
        return [_drop_non_finite(v) for v in value if _is_json_safe(v)]
    return value


class RolloutSample(BaseModel):
    """The conversation + grading artefacts produced by one rollout.

    A rollout produces exactly one sample (one agent run, one reward). There
    used to be a per-sample id so the wire protocol could carry a
    ``dict[str, RolloutSample]``; with the URL-routed single-sample wire
    protocol the id is gone and callers identify rollouts via the URL paths
    they hand the SDK.
    """

    # ``reward`` is the one field the controller cannot do without: delivering
    # it as JSON ``null`` (or failing to encode it at all) leaves the caller
    # waiting on a reward that never arrives. ``validate_assignment`` is the
    # half that matters in practice — graders reach this model through
    # ``GraderContext.set_reward()``, which is a plain attribute assignment.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        validate_assignment=True,
    )

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

    def drop_non_finite_values(self) -> Self:
        """Return a copy whose ``metrics``/``extra_fields`` are JSON-encodable.

        ``allow_inf_nan`` only guards typed float fields, and these two are
        ``Any`` maps that callers routinely mutate in place — which no pydantic
        validator can intercept. Sanitize at the wire boundary instead, and
        drop rather than fail: non-finite telemetry should not cost a rollout
        its reward (same trade-off as ``container/runner.py``).
        """
        if not (_has_non_finite(self.metrics) or _has_non_finite(self.extra_fields)):
            return self
        logger.warning(
            "Dropped non-finite values from sample metrics/extra_fields; "
            "they have no JSON representation"
        )
        return self.model_copy(
            update={
                "metrics": _drop_non_finite(self.metrics),
                "extra_fields": _drop_non_finite(self.extra_fields),
            }
        )


class RolloutStatus(StrEnum):
    """One vocabulary for rollout status everywhere: lifecycle states while
    in flight (status polling), terminal states as execution outcomes."""

    QUEUED = "queued"
    RUNNING = "running"
    GRADING = "grading"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


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
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
    # Backend diagnostics (failure phase, timings); not part of the wire protocol.
    extra_fields: dict[str, Any] | None = None
