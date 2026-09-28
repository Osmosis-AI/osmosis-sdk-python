"""Exercise real LiteLLM conversion against an in-memory provider transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from osmosis_ai.eval.local.llm_bridge import LiteLLMBridge, create_bridge_router


@pytest.fixture
async def upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[httpx.Request]]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if body.get("reject_custom"):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "reject_custom is unsupported; key=wire-test-secret",
                        "type": "invalid_request_error",
                    }
                },
            )
        if request.url.path.endswith("/messages"):
            content = (
                [
                    {"type": "thinking", "thinking": "Use lookup.", "signature": "sig"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
                ]
                if len(requests) == 1
                else [{"type": "text", "text": "done"}]
            )
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": content,
                    "stop_reason": "tool_use" if len(requests) == 1 else "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "reasoning_content": "A short thought.",
                            "reasoning_details": [
                                {"type": "reasoning.encrypted", "data": "opaque"}
                            ],
                            **(
                                {
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                }
                                if body.get("tools")
                                else {}
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(AsyncHTTPHandler, "create_client", lambda self, **kw: client)
    original = litellm.acompletion

    async def complete(**kwargs: Any) -> Any:
        return await original(**kwargs, client=AsyncHTTPHandler())

    monkeypatch.setattr(litellm, "acompletion", complete)
    try:
        yield requests
    finally:
        await client.aclose()


async def test_custom_parameters_reach_openrouter_wire(
    upstream: list[httpx.Request],
) -> None:
    bridge = LiteLLMBridge(
        model="openrouter/openai/gpt-4.1",
        api_base="https://router.example/v1",
        api_key="wire-test-secret",
    )
    app = FastAPI()
    app.include_router(create_bridge_router(bridge, auth_token="test-bearer"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://bridge"
    ) as client:
        response = await client.post(
            "/v1/rollouts/test/chat/completions",
            headers={"Authorization": "Bearer test-bearer"},
            json={
                "model": "client-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.4,
                "custom_option": {"enabled": True, "top_only": 1},
                "reasoning": {"effort": "low"},
                "extra_body": {
                    "custom_option": {"enabled": False},
                    "provider": {"order": ["test-provider"]},
                },
            },
        )
    assert response.status_code == 200
    (request,) = upstream
    assert request.headers["authorization"] == "Bearer wire-test-secret"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4.1"
    assert body["custom_option"] == {"enabled": False, "top_only": 1}
    assert body["temperature"] == 0.4
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {"order": ["test-provider"]}
    assert "drop_params" not in body
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "A short thought."
    assert message["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]


async def test_native_anthropic_tool_loop_preserves_signed_thinking(
    upstream: list[httpx.Request],
) -> None:
    bridge = LiteLLMBridge(
        model="anthropic/claude-sonnet-4-6",
        api_base="https://anthropic.example",
        api_key="wire-test-secret",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    body = {
        "messages": [{"role": "user", "content": "lookup"}],
        "tools": [tool],
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
    }
    response = await bridge.complete(body, rollout_id="test")
    message = response.choices[0].message
    # Simulate a framework dropping extension fields but retaining tool calls.
    await bridge.complete(
        {
            **body,
            "messages": [
                *body["messages"],
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in message.tool_calls],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ],
        },
        rollout_id="test",
    )
    first, second = (json.loads(request.content) for request in upstream)
    assert first["thinking"] == body["thinking"]
    assert first["context_management"] == body["context_management"]
    assistant = next(
        message for message in second["messages"] if message["role"] == "assistant"
    )
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "Use lookup.",
        "signature": "sig",
    }
    assert assistant["content"][1]["type"] == "tool_use"


async def test_openrouter_tool_loop_replays_provider_reasoning_details(
    upstream: list[httpx.Request],
) -> None:
    bridge = LiteLLMBridge(
        model="openrouter/openai/gpt-4.1",
        api_base="https://router.example/v1",
        api_key="wire-test-secret",
    )
    body = {
        "messages": [{"role": "user", "content": "lookup"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    response = await bridge.complete(body, rollout_id="test")
    message = response.choices[0].message
    await bridge.complete(
        {
            **body,
            "messages": [
                *body["messages"],
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in message.tool_calls],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ],
        },
        rollout_id="test",
    )
    assistant = json.loads(upstream[1].content)["messages"][1]
    assert assistant["reasoning_content"] == "A short thought."
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]


async def test_provider_rejection_returns_400_without_retry_or_credentials(
    upstream: list[httpx.Request],
    caplog: pytest.LogCaptureFixture,
) -> None:
    bridge = LiteLLMBridge(
        model="openrouter/openai/gpt-4.1",
        api_base="https://router.example/v1",
        api_key="wire-test-secret",
    )
    app = FastAPI()
    app.include_router(create_bridge_router(bridge, auth_token="test-bearer"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://bridge"
    ) as client:
        response = await client.post(
            "/v1/rollouts/test/chat/completions",
            headers={"Authorization": "Bearer test-bearer"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "reject_custom": True,
            },
        )
    assert response.status_code == 400
    assert len(upstream) == 1
    assert "reject_custom is unsupported" in response.json()["error"]["message"]
    assert "wire-test-secret" not in response.text
    assert "wire-test-secret" not in caplog.text
