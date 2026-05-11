from __future__ import annotations

import json
from typing import Any


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(value)


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not content:
            return ""
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    return text
        return _compact_json(content[0])
    return str(content)


def preprocess_controller_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "unknown"))
        normalized: dict[str, Any] = {
            "role": role,
            "content": _coerce_content(message.get("content")),
        }
        if role == "assistant":
            for key in ("tool_calls", "function_call"):
                if message.get(key) is not None:
                    normalized[key] = message[key]
        elif role == "tool" and message.get("tool_call_id") is not None:
            normalized["tool_call_id"] = message["tool_call_id"]
        elif role == "function" and message.get("name") is not None:
            normalized["name"] = message["name"]
        processed.append(normalized)
    return processed
