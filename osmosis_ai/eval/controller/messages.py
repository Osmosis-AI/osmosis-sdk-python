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
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return item["text"]
        return _compact_json(content[0])
    return str(content)


def preprocess_controller_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    processed: list[dict[str, str]] = []
    for message in messages:
        processed.append(
            {
                "role": str(message.get("role", "unknown")),
                "content": _coerce_content(message.get("content")),
            }
        )
    return processed
