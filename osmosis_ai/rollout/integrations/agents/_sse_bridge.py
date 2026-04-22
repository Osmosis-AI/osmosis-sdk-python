"""SSE→JSON bridge transport for the Chat Completions endpoint.

The Osmosis rollout ``/chat/completions`` endpoint responds with Server-Sent
Events: heartbeat comments plus one or more ``data:`` events carrying either a
complete ``ChatCompletion`` or streamed ``ChatCompletionChunk`` payloads,
followed by ``[DONE]``. ``openai-agents`` drives the endpoint in non-streaming
mode and expects an ``application/json`` body, so this ``httpx`` transport:

1. Injects ``stream=true`` into outbound ``POST /chat/completions`` bodies
   (the controller's non-streaming path is not guaranteed to be correct).
2. Decodes the SSE response with ``openai._streaming.SSEDecoder`` — the
   same decoder openai-python uses for streaming Chat Completions, and
   transitively the one ``litellm`` (hence the Strands integration) uses —
   so spec edge cases (CRLF/CR/LF boundaries, multi-line ``data:``,
   comment frames) come for free.
3. Rewraps the SSE payloads as one ``application/json`` Chat Completion body.

Error semantics match openai-python's streaming layer (``openai/_streaming.py``):
any ``data:`` event whose JSON has a truthy top-level ``error`` field is
translated to HTTP 502 with the upstream message.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from openai._streaming import SSEDecoder
from openai.lib.streaming._deltas import accumulate_delta


def _json_error(
    status: int, message: str, extensions: dict | None = None
) -> httpx.Response:
    body = json.dumps({"error": message}).encode("utf-8")
    return httpx.Response(
        status_code=status,
        headers={
            "content-type": "application/json",
            "content-length": str(len(body)),
        },
        content=body,
        extensions=extensions,
    )


def _inject_stream_true(request: httpx.Request) -> httpx.Request:
    raw = bytes(request.content or b"")
    if not raw:
        return request
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return request
    if not isinstance(body, dict) or body.get("stream") is True:
        return request
    body["stream"] = True
    new_raw = json.dumps(body).encode("utf-8")
    headers = httpx.Headers(request.headers)
    headers["content-length"] = str(len(new_raw))
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=headers,
        content=new_raw,
        extensions=request.extensions,
    )


def _error_message(data: dict) -> str:
    err = data["error"]
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and msg:
            return msg
    return "An error occurred during streaming"


def _merge_logprobs(
    current: dict[str, Any] | None, delta: dict[str, Any] | None
) -> dict[str, Any] | None:
    if delta is None:
        return current
    if current is None:
        return delta

    merged = dict(current)
    for key in ("content", "refusal"):
        delta_items = delta.get(key)
        if delta_items is None:
            continue
        current_items = merged.get(key)
        if current_items is None:
            merged[key] = list(delta_items)
        else:
            merged[key] = [*current_items, *delta_items]
    return merged


def _finalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
    }
    function = tool_call.get("function")
    if isinstance(function, dict):
        result["function"] = {
            key: value
            for key, value in function.items()
            if key != "provider_specific_fields" and value is not None
        }
    return {key: value for key, value in result.items() if value is not None}


def _finalize_message(message_delta: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": message_delta.get("role") or "assistant"}
    for key in ("content", "refusal", "audio"):
        if key in message_delta:
            message[key] = message_delta[key]

    function_call = message_delta.get("function_call")
    if isinstance(function_call, dict):
        message["function_call"] = {
            key: value
            for key, value in function_call.items()
            if key != "provider_specific_fields" and value is not None
        }

    tool_calls = message_delta.get("tool_calls")
    if isinstance(tool_calls, list):
        message["tool_calls"] = [
            _finalize_tool_call(tool_call)
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ]

    return message


def _aggregate_chat_completion_chunks(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("controller returned SSE with no data event")

    first = chunks[0]
    completion: dict[str, Any] = {
        "id": first.get("id"),
        "created": first.get("created"),
        "model": first.get("model"),
        "object": "chat.completion",
    }
    if first.get("service_tier") is not None:
        completion["service_tier"] = first["service_tier"]
    if first.get("system_fingerprint") is not None:
        completion["system_fingerprint"] = first["system_fingerprint"]

    choice_states: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]

        if chunk.get("service_tier") is not None:
            completion["service_tier"] = chunk["service_tier"]
        if chunk.get("system_fingerprint") is not None:
            completion["system_fingerprint"] = chunk["system_fingerprint"]

        for raw_choice in chunk.get("choices", []):
            if not isinstance(raw_choice, dict):
                continue
            index = raw_choice.get("index")
            if not isinstance(index, int):
                raise ValueError("chat.completion.chunk choice missing integer index")

            state = choice_states.setdefault(
                index,
                {
                    "index": index,
                    "finish_reason": None,
                    "message_delta": {},
                    "logprobs": None,
                },
            )

            delta = raw_choice.get("delta")
            if isinstance(delta, dict):
                cleaned_delta = {
                    key: value
                    for key, value in delta.items()
                    if key != "provider_specific_fields"
                }
                state["message_delta"] = accumulate_delta(
                    dict(state["message_delta"]),
                    cleaned_delta,
                )

            finish_reason = raw_choice.get("finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = finish_reason

            state["logprobs"] = _merge_logprobs(
                state["logprobs"],
                raw_choice.get("logprobs"),
            )

    if not choice_states:
        raise ValueError("controller returned chunk stream without any choices")

    choices: list[dict[str, Any]] = []
    for index in sorted(choice_states):
        state = choice_states[index]
        finish_reason = state["finish_reason"]
        if finish_reason is None:
            raise ValueError(
                "controller returned chat.completion.chunk without final finish_reason"
            )

        choice = {
            "index": index,
            "finish_reason": finish_reason,
            "message": _finalize_message(state["message_delta"]),
        }
        if state["logprobs"] is not None:
            choice["logprobs"] = state["logprobs"]
        choices.append(choice)

    completion["choices"] = choices
    if usage is not None:
        completion["usage"] = usage
    return completion


class SSEToJSONBridgeTransport(httpx.AsyncBaseTransport):
    """httpx transport that presents an SSE chat-completions response as JSON."""

    def __init__(self, upstream: httpx.AsyncBaseTransport | None = None) -> None:
        self._upstream = upstream or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        intercept = request.method == "POST" and "/chat/completions" in request.url.path
        if intercept:
            request = _inject_stream_true(request)

        response = await self._upstream.handle_async_request(request)
        if not intercept:
            return response
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response

        await response.aread()

        chunk_payloads: list[dict[str, Any]] = []
        completion_payloads: list[str] = []
        raw_payloads: list[str] = []
        for sse in SSEDecoder().iter_bytes(iter([response.content])):
            if not sse.data or sse.data.startswith("[DONE]"):
                continue
            try:
                data = json.loads(sse.data)
            except json.JSONDecodeError:
                raw_payloads.append(sse.data)
                continue
            if isinstance(data, dict) and data.get("error"):
                return _json_error(502, _error_message(data), response.extensions)
            if isinstance(data, dict) and data.get("object") == "chat.completion.chunk":
                chunk_payloads.append(data)
            elif isinstance(data, dict) and data.get("object") == "chat.completion":
                completion_payloads.append(sse.data)
            else:
                raw_payloads.append(sse.data)

        if chunk_payloads and (completion_payloads or raw_payloads):
            return _json_error(
                502,
                "SSE bridge received mixed chunk and non-chunk payloads",
                response.extensions,
            )

        if chunk_payloads:
            try:
                body = json.dumps(
                    _aggregate_chat_completion_chunks(chunk_payloads)
                ).encode("utf-8")
            except ValueError as exc:
                return _json_error(502, str(exc), response.extensions)
        elif completion_payloads:
            if len(completion_payloads) != 1:
                return _json_error(
                    502,
                    (
                        "controller returned SSE with no data event"
                        if not completion_payloads
                        else "SSE bridge expected exactly one completion payload, "
                        f"got {len(completion_payloads)}"
                    ),
                    response.extensions,
                )
            body = completion_payloads[0].encode("utf-8")
        else:
            if len(raw_payloads) != 1:
                return _json_error(
                    502,
                    (
                        "controller returned SSE with no data event"
                        if not raw_payloads
                        else f"SSE bridge expected exactly one data event, got {len(raw_payloads)}"
                    ),
                    response.extensions,
                )
            body = raw_payloads[0].encode("utf-8")

        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in {"content-type", "content-length", "transfer-encoding"}
        }
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(body))
        return httpx.Response(
            status_code=response.status_code,
            headers=headers,
            content=body,
            extensions=response.extensions,
        )
