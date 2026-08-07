"""Convert ``RolloutSample.trajectory_messages`` (OpenAI chat shape) into ATIF
trajectories. Anything that does not fit the spec losslessly is preserved
under ``extra``.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError

from osmosis_ai.consts import PACKAGE_VERSION
from osmosis_ai.rollout.trajectory.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from osmosis_ai.rollout.trajectory.report import LlmCallMetrics, SampleReport
from osmosis_ai.rollout.types import RolloutSample

logger: logging.Logger = logging.getLogger(__name__)

ATIF_PRODUCER_NAME = "osmosis-rollout-sdk"

# Tool messages fold into the preceding agent step's observation.
_ROLE_TO_SOURCE: dict[str, Literal["system", "user", "agent"]] = {
    "system": "system",
    "developer": "system",
    "assistant": "agent",
    "user": "user",
}


def convert_sample_to_trajectory(
    sample: RolloutSample,
    *,
    rollout_id: str,
    request_label: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    request_extra_fields: dict[str, Any] | None = None,
    report: SampleReport | None = None,
    default_model_name: str | None = None,
    unmatched_sample_reports: Mapping[str, SampleReport] | None = None,
) -> Trajectory:
    """Build the ATIF trajectory document for the rollout's single sample.

    ``report`` carries controller-reported per-call metrics (see
    ``report.py``); ``unmatched_sample_reports`` are entries that could
    not be attributed to the sample, preserved under ``extra``.
    """
    if sample.trajectory_messages is None:
        raise ValueError("Sample has no trajectory-compatible messages")
    steps = _messages_to_steps(sample.trajectory_messages)
    unmatched_llm_call_metrics = _apply_report(steps, report)
    final_metrics = _final_metrics_from_report(report) or FinalMetrics()
    final_metrics.total_steps = len(steps)
    model_name = (report.model_name if report else None) or default_model_name
    unmatched_reports: dict[str, Any] | None = None
    if unmatched_sample_reports:
        unmatched_reports = {
            key: value.model_dump(exclude_none=True)
            for key, value in unmatched_sample_reports.items()
        }
    return Trajectory(
        session_id=rollout_id,
        trajectory_id=rollout_id,
        agent=Agent(
            name=ATIF_PRODUCER_NAME, version=PACKAGE_VERSION, model_name=model_name
        ),
        steps=steps,
        final_metrics=final_metrics,
        extra=_compose_extra(
            sample,
            rollout_id=rollout_id,
            request_label=request_label,
            request_metadata=request_metadata,
            request_extra_fields=request_extra_fields,
            unmatched_llm_call_metrics=unmatched_llm_call_metrics,
            unmatched_sample_reports=unmatched_reports,
        ),
    )


def _messages_to_steps(messages: Sequence[Mapping[str, Any]]) -> list[Step]:
    steps: list[Step] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role in ("tool", "function"):
            _attach_tool_result(steps, message)
            continue

        source = _ROLE_TO_SOURCE.get(role)
        extra: dict[str, Any] | None = None
        if source is None:
            # Unknown role: keep the turn, preserve the original role.
            source = "user"
            extra = {"original_role": role}

        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            reasoning_content = json.dumps(
                reasoning_content, ensure_ascii=False, default=str
            )
        tool_calls = _convert_tool_calls(message.get("tool_calls"))
        if source != "agent" and (reasoning_content is not None or tool_calls):
            # ATIF allows these only on agent steps; demote to extra so one
            # odd message cannot fail the whole document's validation.
            extra = dict(extra or {})
            if reasoning_content is not None:
                extra["reasoning_content"] = reasoning_content
            if tool_calls:
                extra["tool_calls"] = [
                    tc.model_dump(exclude_none=True) for tc in tool_calls
                ]
            reasoning_content = None
            tool_calls = None

        metrics = _metrics_from_message(message) if source == "agent" else None
        steps.append(
            Step(
                step_id=len(steps) + 1,
                timestamp=_message_timestamp(message),
                source=source,
                message=_coerce_content(message.get("content")),
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                model_name=_message_model_name(message) if source == "agent" else None,
                metrics=metrics,
                llm_call_count=1 if metrics else None,
                extra=extra,
            )
        )
    return steps


def _iso_timestamp(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=UTC).isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    return None


def _message_timestamp(message: Mapping[str, Any]) -> str | None:
    """Best upstream timestamp on the message (harbor's mini-swe convention)."""
    for field in ("created_at", "timestamp", "completed_at"):
        timestamp = _iso_timestamp(message.get(field))
        if timestamp:
            return timestamp
    extra = message.get("extra")
    if isinstance(extra, Mapping):
        return _iso_timestamp(extra.get("timestamp"))
    return None


def _message_response(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Raw completion response under ``extra.response`` (harbor convention)."""
    extra = message.get("extra")
    response = extra.get("response") if isinstance(extra, Mapping) else None
    return response if isinstance(response, Mapping) else None


def _message_model_name(message: Mapping[str, Any]) -> str | None:
    model = message.get("model") or message.get("model_name")
    if not model:
        response = _message_response(message)
        model = response.get("model") if response else None
    return str(model) if model else None


def _metrics_from_message(message: Mapping[str, Any]) -> Metrics | None:
    """Read usage stashed on the message (top-level ``usage`` or harbor's
    ``extra.response``); accepts chat-completions and Responses API names."""
    usage = message.get("usage")
    if not isinstance(usage, Mapping):
        response = _message_response(message)
        usage = response.get("usage") if response else None
        if not isinstance(usage, Mapping):
            return None

    prompt_details = usage.get("prompt_tokens_details") or usage.get(
        "input_tokens_details"
    )
    cached = usage.get("cached_tokens")
    if cached is None and isinstance(prompt_details, Mapping):
        cached = prompt_details.get("cached_tokens")

    extra: dict[str, Any] = {}
    for key in (
        "prompt_tokens_details",
        "input_tokens_details",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        details = usage.get(key)
        if isinstance(details, Mapping) and details:
            extra[key] = dict(details)

    # `is None` (not `or`): a genuine 0 must not fall through to the other
    # API's field name.
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens")

    try:
        metrics = Metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            cost_usd=usage.get("cost_usd"),
            extra=extra or None,
        )
    except ValidationError:
        return None
    return metrics if metrics.model_dump(exclude_none=True) else None


def _apply_report(
    steps: list[Step], report: SampleReport | None
) -> list[dict[str, Any]] | None:
    """Overlay per-call metrics onto agent steps in dispatch order.

    Only when the counts match exactly; otherwise the metrics are returned
    for preservation under ``extra`` rather than mis-attributed.
    """
    if report is None or not report.llm_call_metrics:
        return None
    agent_steps = [step for step in steps if step.source == "agent"]
    if len(agent_steps) != len(report.llm_call_metrics):
        return [e.model_dump(exclude_none=True) for e in report.llm_call_metrics]
    for step, entry in zip(agent_steps, report.llm_call_metrics, strict=True):
        reported = _metrics_from_report_entry(entry)
        if reported is not None:
            # Merge field by field: the count-only report must not erase
            # native token ids and logprobs.
            step.metrics = (
                reported
                if step.metrics is None
                else step.metrics.model_copy(
                    update=reported.model_dump(exclude_none=True)
                )
            )
        step.model_name = entry.model_name or step.model_name
        step.llm_call_count = 1
    return None


def _metrics_from_report_entry(entry: LlmCallMetrics) -> Metrics | None:
    # model_name is the one report field that lives on the Step, not in
    # Metrics. Mechanical dump keeps the schemas in lockstep: a field added
    # to LlmCallMetrics but missing from Metrics fails loudly (extra=forbid)
    # instead of being silently dropped.
    payload = entry.model_dump(exclude={"model_name"}, exclude_none=True)
    if not payload:
        return None
    return Metrics.model_validate(payload)


def _final_metrics_from_report(report: SampleReport | None) -> FinalMetrics | None:
    """Controller-provided totals win; otherwise sum the per-call metrics."""
    if report is None:
        return None
    if report.final_metrics:
        payload = dict(report.final_metrics)
        payload.pop("total_steps", None)
        if payload:
            try:
                return FinalMetrics.model_validate(payload)
            except ValidationError:
                logger.warning(
                    "Ignoring malformed final_metrics in trajectory report",
                    exc_info=True,
                )
    if not report.llm_call_metrics:
        return None
    entries = report.llm_call_metrics
    prompt = [e.prompt_tokens for e in entries if e.prompt_tokens is not None]
    completion = [
        e.completion_tokens for e in entries if e.completion_tokens is not None
    ]
    cached = [e.cached_tokens for e in entries if e.cached_tokens is not None]
    cost = [e.cost_usd for e in entries if e.cost_usd is not None]
    totals = FinalMetrics(
        total_prompt_tokens=sum(prompt) if prompt else None,
        total_completion_tokens=sum(completion) if completion else None,
        total_cached_tokens=sum(cached) if cached else None,
        total_cost_usd=sum(cost) if cost else None,
    )
    return totals if totals.model_dump(exclude_none=True) else None


def _attach_tool_result(steps: list[Step], message: Mapping[str, Any]) -> None:
    """Fold an OpenAI ``tool`` message into the preceding agent step."""
    # ATIF requires source_call_id to match a tool call in the same step.
    call_id = message.get("tool_call_id")
    target = steps[-1] if steps and steps[-1].source == "agent" else None
    known_ids = (
        {tc.tool_call_id for tc in target.tool_calls}
        if target and target.tool_calls
        else set()
    )
    matched = call_id in known_ids

    result = ObservationResult(
        source_call_id=call_id if matched else None,
        content=_coerce_content(message.get("content")),
        extra=None if matched else {"tool_call_id": call_id},
    )

    if target is None:
        # Tool result with no preceding agent turn: record as system-observed.
        steps.append(
            Step(
                step_id=len(steps) + 1,
                source="system",
                message="",
                observation=Observation(results=[result]),
                extra={"original_role": str(message.get("role", "tool"))},
            )
        )
        return

    if target.observation is None:
        target.observation = Observation(results=[result])
    else:
        target.observation.results.append(result)


def _convert_tool_calls(raw: Any) -> list[ToolCall] | None:
    # Best-effort: a malformed tool_calls value must not fail the document.
    if not raw or isinstance(raw, str) or not isinstance(raw, Sequence):
        return None
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        function = function if isinstance(function, Mapping) else {}
        calls.append(
            ToolCall(
                tool_call_id=str(item.get("id") or f"call_{index}"),
                function_name=str(function.get("name") or "unknown"),
                arguments=_coerce_arguments(function.get("arguments")),
            )
        )
    return calls or None


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"_raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": raw}
    return {}


def _coerce_content(content: Any) -> str:
    """Flatten OpenAI message content to text; non-text parts become JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = [p for p in content if isinstance(p, Mapping)]
        if parts and all(p.get("type") == "text" for p in parts):
            return "\n".join(str(p.get("text", "")) for p in parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def _compose_extra(
    sample: RolloutSample,
    *,
    rollout_id: str,
    request_label: str | None,
    request_metadata: dict[str, Any] | None,
    request_extra_fields: dict[str, Any] | None,
    unmatched_llm_call_metrics: list[dict[str, Any]] | None,
    unmatched_sample_reports: dict[str, Any] | None,
) -> dict[str, Any]:
    """Namespace all Osmosis platform context under ``extra["osmosis"]``."""
    osmosis: dict[str, Any] = {
        "rollout_id": rollout_id,
        "label": sample.label if sample.label is not None else request_label,
        "reward": sample.reward,
        "sample_metrics": sample.metrics or None,
        "sample_extra_fields": sample.extra_fields or None,
        "request_metadata": request_metadata,
        "request_extra_fields": request_extra_fields,
        "unmatched_llm_call_metrics": unmatched_llm_call_metrics,
        "unmatched_sample_reports": unmatched_sample_reports,
    }
    return {"osmosis": {k: v for k, v in osmosis.items() if v is not None}}
