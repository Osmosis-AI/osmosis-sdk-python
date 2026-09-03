import copy
import logging
import math
import numbers
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from osmosis_ai.rollout.utils.identifiers import ensure_single_path_segment

logger: logging.Logger = logging.getLogger(__name__)

MessageDict = dict[str, Any]
SampleMessage = Mapping[str, Any]

# Sentinel for values with no JSON representation at all.
_DROP = object()


def _json_ready_scalar(value: Any) -> bool:
    """A scalar ``json.dumps`` encodes as-is, no normalization needed.

    Exact builtin types only: NumPy's ``float64`` is a ``float`` subclass, so
    an ``isinstance`` check would pass it through untouched and leave a foreign
    scalar in the ``Any`` telemetry map. Non-finite floats are excluded too —
    ``json.dumps`` emits the non-standard ``NaN``/``Infinity`` literals.
    """
    if value is None or isinstance(value, (bool, str)):
        return True
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _needs_json_fit(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(k, str) or _needs_json_fit(v) for k, v in value.items()
        )
    # Tuples encode as JSON arrays too, so they carry the same hazards.
    if isinstance(value, (list, tuple)):
        return any(_needs_json_fit(v) for v in value)
    return not _json_ready_scalar(value)


def _json_fit(value: Any, dropped: list[str]) -> Any:
    """Return *value* as something ``json.dumps`` encodes, or ``_DROP``.

    Numeric scalars outside the builtins — NumPy's, most commonly — normalize
    to built-in int/float through the ``numbers`` ABCs, which numpy registers
    its scalar types with; the SDK never has to import numpy itself. Non-finite
    numbers and everything else JSON cannot represent drop, recorded in
    *dropped* for one summary warning.
    """
    if _json_ready_scalar(value):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if math.isfinite(as_float):
            return as_float
        dropped.append(f"non-finite {type(value).__name__}")
        return _DROP
    if isinstance(value, dict):
        fitted: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str):
                key = k
            elif isinstance(k, bool) or not isinstance(k, numbers.Real):
                dropped.append(f"unencodable key {type(k).__name__}")
                continue
            else:
                # JSON keys are strings; stringify numeric keys the same way
                # json.dumps would have.
                num = _json_fit(k, dropped)
                if num is _DROP:
                    continue
                key = str(num)
            val = _json_fit(v, dropped)
            if val is _DROP:
                continue
            fitted[key] = val
        return fitted
    if isinstance(value, (list, tuple)):
        return [f for v in value if (f := _json_fit(v, dropped)) is not _DROP]
    dropped.append(type(value).__name__)
    return _DROP


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

    def json_safe_copy(self) -> Self:
        """Return a copy whose ``metrics``/``extra_fields`` survive JSON encoding.

        ``allow_inf_nan`` only guards typed float fields, and these two are
        ``Any`` maps that callers routinely mutate in place — which no pydantic
        validator can intercept — and routinely fill with NumPy scalars, which
        ``json.dumps`` rejects outright. Sanitize at the wire boundary instead:
        normalize foreign numeric scalars to built-in int/float, and drop
        non-finite values and unencodable objects rather than fail — telemetry
        should not cost a rollout its reward (same trade-off as
        ``container/runner.py``).
        """
        if not (_needs_json_fit(self.metrics) or _needs_json_fit(self.extra_fields)):
            return self
        dropped: list[str] = []
        fitted = self.model_copy(
            update={
                "metrics": _json_fit(self.metrics, dropped),
                "extra_fields": _json_fit(self.extra_fields, dropped),
            }
        )
        if dropped:
            logger.warning(
                "Dropped values with no JSON representation from sample "
                "metrics/extra_fields: %s",
                ", ".join(sorted(set(dropped))),
            )
        return fitted


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
    LEASE_EXPIRED = "lease_expired"
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
    grade: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        # ``id`` is joined onto host paths by every backend.
        return ensure_single_path_segment(value, label="rollout_id")

    @field_validator("agent_timeout_sec", "grader_timeout_sec")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        # NaN reaches the event loop's selector and crashes it (observed on
        # 3.13), and +/-inf silently disables enforcement — neither is a
        # deadline. "No deadline" is spelled None. Zero and negative values
        # stay allowed and mean an immediately-expired deadline.
        if value is not None and not math.isfinite(value):
            raise ValueError("timeout must be finite; omit it to run unbounded")
        return value


class ExecutionResult(BaseModel):
    status: RolloutStatus
    sample: RolloutSample | None = None
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None
    # Backend diagnostics (failure phase, timings); not part of the wire protocol.
    extra_fields: dict[str, Any] | None = None


class ExecutionOutcome(BaseModel):
    """Workflow result and optional grader result from one execution."""

    workflow: ExecutionResult
    grader: ExecutionResult | None = None

    @property
    def result(self) -> ExecutionResult:
        return self.grader or self.workflow

    @property
    def result_to_save(self) -> ExecutionResult:
        if self.grader is not None and self.grader.sample is not None:
            return self.grader
        return self.workflow
