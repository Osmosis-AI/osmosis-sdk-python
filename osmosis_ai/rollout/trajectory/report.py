"""Optional per-call LLM metrics supplied to trajectory persistence.

Callers can pass reports directly to ``save_trajectory`` or the converter.
The long-polling protocol does not exchange reports; ``report_from_response``
remains available for parsing legacy callback acknowledgements.
"""

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger: logging.Logger = logging.getLogger(__name__)


class LlmCallMetrics(BaseModel):
    """Metrics for one LLM inference, in dispatch order.

    Fields mirror ATIF's ``Metrics`` slots 1:1 (``model_name`` maps onto
    the step's); token ids carry the exact engine tokenization for
    training-grade documents.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    model_name: str | None = None
    extra: dict[str, Any] | None = None


class SampleReport(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    model_name: str | None = None
    llm_call_metrics: list[LlmCallMetrics] = Field(default_factory=list)
    final_metrics: dict[str, Any] | None = None


class TrajectoryReport(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    model_name: str | None = None
    samples: dict[str, SampleReport] = Field(default_factory=dict)


def report_from_response(response: httpx.Response) -> TrajectoryReport | None:
    """Extract the ``trajectory`` object from a callback ack body.

    Best-effort: any body without a well-formed ``trajectory`` yields ``None``.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, Mapping):
        return None
    payload = body.get("trajectory")
    if payload is None:
        return None
    try:
        return TrajectoryReport.model_validate(payload)
    except ValidationError:
        logger.warning(
            "Ignoring malformed trajectory report in callback response",
            exc_info=True,
        )
        return None
