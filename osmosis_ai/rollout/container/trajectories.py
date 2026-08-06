"""Extract chat messages from the trajectory documents agents leave behind.

Shared by the in-container grader runner and the host-side backend, so it
must not import Harbor.
"""

from __future__ import annotations

import json
from typing import Any

ATIF_ROLE_BY_SOURCE = {"user": "user", "agent": "assistant", "system": "system"}


def _chat_tool_calls(raw: Any) -> list[dict[str, Any]] | None:
    """ATIF tool calls in OpenAI chat shape (arguments as a JSON string)."""
    if not isinstance(raw, list):
        return None
    calls = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                "id": call.get("tool_call_id"),
                "type": "function",
                "function": {
                    "name": call.get("function_name"),
                    "arguments": json.dumps(
                        call.get("arguments") or {}, ensure_ascii=False, default=str
                    ),
                },
            }
        )
    return calls or None


def messages_from_trajectory(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept ATIF (steps) or raw harness formats (a messages list).

    Agent steps keep tool calls, reasoning, and observations; dropping them
    would emit consecutive assistant turns that never existed.
    """
    if isinstance(document.get("messages"), list):
        return document["messages"]
    messages: list[dict[str, Any]] = []
    for step in document.get("steps") or []:
        content = step.get("message")
        role = ATIF_ROLE_BY_SOURCE.get(step.get("source"), "tool")
        tool_calls = (
            _chat_tool_calls(step.get("tool_calls")) if role == "assistant" else None
        )
        if content is not None or tool_calls:
            message: dict[str, Any] = {
                "role": role,
                "content": content if content is not None else "",
            }
            reasoning_content = step.get("reasoning_content")
            if role == "assistant" and reasoning_content is not None:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        observation = step.get("observation")
        results = observation.get("results") if isinstance(observation, dict) else None
        for result in results or []:
            if not isinstance(result, dict):
                continue
            tool_message: dict[str, Any] = {
                "role": "tool",
                "content": result.get("content") or "",
            }
            if result.get("source_call_id") is not None:
                tool_message["tool_call_id"] = result["source_call_id"]
            messages.append(tool_message)
    return messages
