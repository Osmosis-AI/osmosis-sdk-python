"""Convert ``RolloutSample`` conversations (OpenAI chat shape) into ATIF
trajectories, using Harbor's reference models for spec validation. Anything
that does not fit the spec losslessly is preserved under ``extra``.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from harbor.models.trajectories import (
    Agent,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

from osmosis_ai.consts import PACKAGE_VERSION
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
    sample_id: str,
    request_label: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    request_extra_fields: dict[str, Any] | None = None,
) -> Trajectory:
    """Build one ATIF trajectory document for one rollout sample."""
    return Trajectory(
        session_id=rollout_id,
        trajectory_id=f"{rollout_id}/{sample_id}",
        agent=Agent(name=ATIF_PRODUCER_NAME, version=PACKAGE_VERSION),
        steps=messages_to_steps(sample.messages),
        extra=_compose_extra(
            sample,
            rollout_id=rollout_id,
            sample_id=sample_id,
            request_label=request_label,
            request_metadata=request_metadata,
            request_extra_fields=request_extra_fields,
        ),
    )


def messages_to_steps(messages: Sequence[Mapping[str, Any]]) -> list[Step]:
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
            # ATIF allows reasoning_content/tool_calls only on agent steps;
            # demote them to extra so one odd message cannot fail the whole
            # document's validation.
            extra = dict(extra or {})
            if reasoning_content is not None:
                extra["reasoning_content"] = reasoning_content
            if tool_calls:
                extra["tool_calls"] = [
                    tc.model_dump(exclude_none=True) for tc in tool_calls
                ]
            reasoning_content = None
            tool_calls = None

        steps.append(
            Step(
                step_id=len(steps) + 1,
                source=source,
                message=_coerce_content(message.get("content")),
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                extra=extra,
            )
        )
    return steps


def _attach_tool_result(steps: list[Step], message: Mapping[str, Any]) -> None:
    """Fold an OpenAI ``tool`` message into the preceding agent step."""
    # ATIF requires source_call_id to match a tool call in the same step;
    # unmatched ids are kept under extra instead.
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
    if not raw:
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
    sample_id: str,
    request_label: str | None,
    request_metadata: dict[str, Any] | None,
    request_extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Namespace all Osmosis platform context under ``extra["osmosis"]``."""
    osmosis: dict[str, Any] = {
        "rollout_id": rollout_id,
        "sample_id": sample_id,
        "label": sample.label if sample.label is not None else request_label,
        "reward": sample.reward,
        "sample_metrics": sample.metrics or None,
        "sample_extra_fields": sample.extra_fields or None,
        "request_metadata": request_metadata,
        "request_extra_fields": request_extra_fields,
    }
    return {"osmosis": {k: v for k, v in osmosis.items() if v is not None}}
