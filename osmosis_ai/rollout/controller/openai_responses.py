"""Chat Completions compatibility adapter for OpenAI's Responses API.

Local rollout clients speak Chat Completions. Official OpenAI models with
function tools need the native Responses API, so this module owns the protocol
translation while :mod:`llm_bridge` remains responsible for HTTP and routing.
"""

from __future__ import annotations

import json
import time
from typing import Any


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


def _convert_content(content: Any, *, role: Any) -> Any:
    if not isinstance(content, list):
        return content

    converted: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append(part)
            continue

        part_type = part.get("type")
        if part_type == "text":
            converted.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": part.get("text", ""),
                }
            )
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": image_url.get("url"),
                        "detail": image_url.get("detail", "auto"),
                    }
                )
            else:
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "auto",
                    }
                )
        elif part_type == "file" and isinstance(part.get("file"), dict):
            file = part["file"]
            converted.append(
                {
                    "type": "input_file",
                    **{
                        key: file[key]
                        for key in ("file_id", "file_data", "filename")
                        if key in file
                    },
                }
            )
        else:
            converted.append(part)
    return converted


def _convert_messages(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []

    items: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role == "tool":
            output = _convert_content(message.get("content", ""), role=role)
            if not isinstance(output, (str, list)):
                output = _compact_json(output)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": output,
                }
            )
            continue

        tool_calls = message.get("tool_calls")
        content = message.get("content")
        refusal = message.get("refusal") if role == "assistant" else None
        if content not in (None, ""):
            items.append(
                {"role": role, "content": _convert_content(content, role=role)}
            )
        elif isinstance(refusal, str) and refusal:
            # A refused turn has null content; dropping the refusal here would
            # replay the conversation with an empty assistant message.
            items.append(
                {"role": role, "content": [{"type": "refusal", "refusal": refusal}]}
            )
        elif not tool_calls:
            items.append({"role": role, "content": ""})

        if role != "assistant" or not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = _compact_json(arguments)
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                }
            )
    return items


def _convert_tools(tools: Any) -> list[Any] | None:
    if not isinstance(tools, list):
        return None

    converted: list[Any] = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
        ):
            converted.append({"type": "function", **tool["function"]})
        else:
            converted.append(tool)
    return converted


def _convert_tool_choice(tool_choice: Any) -> Any:
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
    ):
        return {
            "type": "function",
            "name": tool_choice["function"].get("name", ""),
        }
    return tool_choice


def build_responses_kwargs(
    body: dict[str, Any], *, model: str, api_key: str | None
) -> dict[str, Any]:
    """Build a LiteLLM Responses request from a Chat Completions body."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input": _convert_messages(body.get("messages", [])),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if "max_completion_tokens" in body:
        kwargs["max_output_tokens"] = body["max_completion_tokens"]
    elif "max_tokens" in body:
        kwargs["max_output_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "parallel_tool_calls"):
        if key in body:
            kwargs[key] = body[key]
    if "tool_choice" in body:
        kwargs["tool_choice"] = _convert_tool_choice(body["tool_choice"])
    tools = _convert_tools(body.get("tools"))
    if tools is not None:
        kwargs["tools"] = tools
    return kwargs


def to_chat_response(response: Any) -> dict[str, Any]:
    """Adapt a native Responses result to a Chat Completions response."""
    status = _field(response, "status")
    incomplete_details = _field(response, "incomplete_details")
    incomplete_reason = _field(incomplete_details, "reason")
    if status == "incomplete":
        if incomplete_reason not in {"max_output_tokens", "content_filter"}:
            raise RuntimeError(
                "OpenAI Responses API returned an incomplete response "
                f"with unknown reason {incomplete_reason!r}"
            )
    elif status != "completed":
        error = _field(response, "error")
        error_code = _field(error, "code")
        error_message = _field(error, "message")
        detail = ": ".join(str(value) for value in (error_code, error_message) if value)
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"OpenAI Responses API returned unsuccessful status {status!r}{suffix}"
        )

    text_parts: list[str] = []
    refusal_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in _field(response, "output", []) or []:
        item_type = _field(item, "type")
        if item_type == "message":
            for content in _field(item, "content", []) or []:
                if _field(content, "type") == "output_text":
                    text_parts.append(str(_field(content, "text", "") or ""))
                elif _field(content, "type") == "refusal":
                    refusal_parts.append(str(_field(content, "refusal", "") or ""))
        elif item_type == "function_call":
            call_id = _field(item, "call_id") or _field(item, "id")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": _field(item, "name", ""),
                        "arguments": _field(item, "arguments", ""),
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if refusal_parts:
        message["refusal"] = "".join(refusal_parts)

    finish_reason = "tool_calls" if tool_calls else "stop"
    if status == "incomplete":
        finish_reason = (
            "length" if incomplete_reason == "max_output_tokens" else "content_filter"
        )

    adapted: dict[str, Any] = {
        "id": _field(response, "id", "chatcmpl-eval"),
        "created": int(_field(response, "created_at", time.time()) or time.time()),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    usage = _field(response, "usage")
    if usage is not None:
        input_tokens = int(_field(usage, "input_tokens", 0) or 0)
        output_tokens = int(_field(usage, "output_tokens", 0) or 0)
        adapted["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": int(
                _field(usage, "total_tokens", input_tokens + output_tokens) or 0
            ),
        }
    return adapted


__all__ = [
    "build_responses_kwargs",
    "to_chat_response",
]
