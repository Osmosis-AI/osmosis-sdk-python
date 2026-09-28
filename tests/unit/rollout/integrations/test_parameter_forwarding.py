"""Exercise real framework and LiteLLM serialization at the HTTP boundary."""

from __future__ import annotations

import json

import httpx
import pytest
from agents import ModelSettings, function_tool
from agents.models.interface import ModelTracing
from openai.types.shared import Reasoning

from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.integrations.agents.openai_agents import (
    OsmosisLitellmModel,
    OsmosisMemorySession,
)
from osmosis_ai.rollout.integrations.agents.strands import (
    OsmosisRolloutModel,
    OsmosisStrandsAgent,
)


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []

    async def send(_client, request, **_kwargs):
        assert request.url == "http://rollout.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        requests.append(body)
        response = {
            "id": "completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": "osmosis-rollout",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response, request=request)
        chunks = []
        for delta, finish_reason in [
            ({"role": "assistant", "content": "hello"}, None),
            ({}, "stop"),
        ]:
            chunk = {
                **response,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return requests


@pytest.fixture
def rollout_context():
    with RolloutContext(
        chat_completions_url="http://rollout.test/v1",
        api_key="test-key",
        rollout_id="parameter-forwarding",
    ) as ctx:
        yield ctx


@function_tool
def weather(city: str) -> str:
    """Get the weather for a city."""
    return city


def assert_forwarded_parameters(body):
    assert body["model"] == "osmosis-rollout"
    assert body["reasoning_effort"] == "low"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body["user"] == "test-user"
    assert body["provider"] == {"sort": "latency"}
    assert body["custom_parameter"] == {"enabled": True}
    assert body["custom_top_level"] == {"enabled": True}
    assert body["context_management"] == {"edits": []}
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    # Explicit extra_body values keep the OpenAI client's override semantics.
    assert body["temperature"] == 0.1
    assert body["frequency_penalty"] == 0.2
    assert body["tools"][0]["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("response_method", ["get_response", "stream_response"])
async def test_agents_response_methods_forward_parameters_over_streaming_transport(
    rollout_context, captured_requests, response_method
):
    OsmosisMemorySession()
    model = OsmosisLitellmModel()
    kwargs = {
        "system_instructions": None,
        "input": "hello",
        "model_settings": ModelSettings(
            temperature=0.9,
            top_logprobs=2,
            reasoning=Reasoning(effort="low"),
            extra_args={
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "user": "test-user",
                "custom_top_level": {"enabled": True},
            },
            extra_body={
                "provider": {"sort": "latency"},
                "custom_parameter": {"enabled": True},
                "context_management": {"edits": []},
                "temperature": 0.1,
                "frequency_penalty": 0.2,
            },
        ),
        "tools": [weather],
        "output_schema": None,
        "handoffs": [],
        "tracing": ModelTracing.DISABLED,
    }

    if response_method == "stream_response":
        events = [event async for event in model.stream_response(**kwargs)]
        response = next(
            event.response for event in events if event.type == "response.completed"
        )
    else:
        response = await model.get_response(**kwargs)

    assert response.output[0].content[0].text == "hello"
    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert_forwarded_parameters(body)
    # Both model APIs use SSE; get_response aggregates it for the caller.
    assert body["stream"] is True
    assert body["tools"][0]["function"]["strict"] is True


@pytest.mark.parametrize("stream", [False, True])
async def test_strands_forward_parameters_to_gateway(
    rollout_context, captured_requests, stream
):
    agent = OsmosisStrandsAgent(
        model=OsmosisRolloutModel(
            params={
                "stream": stream,
                "temperature": 0.9,
                "logprobs": True,
                "top_logprobs": 2,
                "reasoning_effort": "low",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "user": "test-user",
                "custom_top_level": {"enabled": True},
                "extra_body": {
                    "provider": {"sort": "latency"},
                    "custom_parameter": {"enabled": True},
                    "context_management": {"edits": []},
                    "temperature": 0.1,
                    "frequency_penalty": 0.2,
                },
            }
        ),
        callback_handler=None,
    )
    events = [
        event
        async for event in agent.model.stream(
            [{"role": "user", "content": [{"text": "hello"}]}],
            tool_specs=[
                {
                    "name": "weather",
                    "description": "Get the weather for a city.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        }
                    },
                }
            ],
        )
    ]

    assert any(
        event.get("contentBlockDelta", {}).get("delta", {}).get("text") == "hello"
        for event in events
    )
    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert_forwarded_parameters(body)
    assert body.get("stream", False) is stream
