"""Tests for osmosis_ai.rollout.integrations.agents._sse_bridge."""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest

from osmosis_ai.rollout.integrations.agents._sse_bridge import (
    SSEToJSONBridgeTransport,
)


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Upstream test double that records the request and returns a canned
    response, so we can assert on request mutation + response rewrapping."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.received: httpx.Request | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.received = request
        return self.response


def _sse_response(body_text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"content-type": "text/event-stream"},
        content=body_text.encode("utf-8"),
    )


def _chat_completions_request(body: dict) -> httpx.Request:
    raw = json.dumps(body).encode("utf-8")
    return httpx.Request(
        method="POST",
        url="http://controller.local/chat/completions",
        headers={"content-type": "application/json", "content-length": str(len(raw))},
        content=raw,
    )


def _chunk_payload(
    *,
    content: str | None = None,
    finish_reason: str | None = "stop",
    role: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if role is not None:
        delta["role"] = role
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    choice = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "osmosis-rollout",
        "choices": [choice],
        "usage": usage,
    }


class TestSSEToJSONBridgeTransport:
    async def test_forwards_non_chat_completions_unchanged(self):
        health = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"status":"ok"}',
        )
        upstream = _RecordingTransport(health)
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        request = httpx.Request(method="GET", url="http://controller.local/health")
        response = await transport.handle_async_request(request)

        assert response is health
        assert upstream.received is request

    async def test_injects_stream_true_on_chat_completions(self):
        sse = _sse_response('data: {"id":"c1","choices":[]}\n\ndata: [DONE]\n\n')
        upstream = _RecordingTransport(sse)
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        request = _chat_completions_request({"model": "x", "messages": []})
        await transport.handle_async_request(request)

        forwarded_body = json.loads(upstream.received.content)
        assert forwarded_body["stream"] is True

    async def test_preserves_existing_stream_true(self):
        sse = _sse_response('data: {"id":"c1","choices":[]}\n\ndata: [DONE]\n\n')
        upstream = _RecordingTransport(sse)
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        request = _chat_completions_request(
            {"model": "x", "messages": [], "stream": True}
        )
        await transport.handle_async_request(request)

        forwarded_body = json.loads(upstream.received.content)
        assert forwarded_body["stream"] is True

    async def test_rewraps_sse_as_json(self):
        payload = {"id": "c1", "object": "chat.completion", "choices": [{"x": 1}]}
        sse_body = f": ping\n\ndata: {json.dumps(payload)}\n\ndata: [DONE]\n\n"
        upstream = _RecordingTransport(_sse_response(sse_body))
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.headers["content-type"] == "application/json"
        assert json.loads(response.content) == payload

    async def test_errors_when_no_data_event(self):
        upstream = _RecordingTransport(_sse_response(": ping\n\n"))
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 502
        assert "no data event" in json.loads(response.content)["error"]

    async def test_errors_when_multiple_data_events(self):
        body = 'data: {"a":1}\n\ndata: {"b":2}\n\n'
        upstream = _RecordingTransport(_sse_response(body))
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 502
        assert "exactly one data event" in json.loads(response.content)["error"]

    async def test_passes_through_non_sse_chat_completions_response(self):
        """The bridge should not interfere when the upstream already returns
        application/json."""
        json_resp = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"id":"c1"}',
        )
        upstream = _RecordingTransport(json_resp)
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response is json_resp

    async def test_rewraps_real_sse_starlette_crlf_body(self):
        """Reproduces the end-to-end wire path against bytes shaped like
        sse_starlette's actual output (CRLF line endings + CRLF-CRLF event
        boundaries + a heartbeat comment)."""
        chunk = _chunk_payload(
            content="hi",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        body = (
            b": ping\r\n\r\n"
            b"data: " + json.dumps(chunk).encode() + b"\r\n\r\n"
            b"data: [DONE]\r\n\r\n"
        )
        upstream = _RecordingTransport(
            httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        )
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert json.loads(response.content) == {
            "id": chunk["id"],
            "object": "chat.completion",
            "created": chunk["created"],
            "model": chunk["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def test_aggregates_multiple_chunk_events_into_one_completion(self):
        first = _chunk_payload(content="hel", finish_reason=None, role="assistant")
        second = {
            **_chunk_payload(
                content="lo",
                finish_reason="stop",
                usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            ),
            "id": first["id"],
            "created": first["created"],
        }
        upstream = _RecordingTransport(
            _sse_response(
                "".join(
                    [
                        f"data: {json.dumps(first)}\n\n",
                        f"data: {json.dumps(second)}\n\n",
                        "data: [DONE]\n\n",
                    ]
                )
            )
        )
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 200
        assert json.loads(response.content) == {
            "id": first["id"],
            "object": "chat.completion",
            "created": first["created"],
            "model": "osmosis-rollout",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

    async def test_rewraps_chunk_tool_calls_as_completion(self):
        chunk = _chunk_payload(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a":1,"b":2}',
                    },
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
        upstream = _RecordingTransport(
            _sse_response(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        )
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 200
        assert json.loads(response.content) == {
            "id": chunk["id"],
            "object": "chat.completion",
            "created": chunk["created"],
            "model": chunk["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": '{"a":1,"b":2}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    async def test_translates_sse_error_envelope_to_http_error(self):
        """An SSE data event carrying ``{"error": "..."}`` must surface as
        an HTTP 5xx error, matching openai-python's streaming behavior."""
        body = (
            b": ping\r\n\r\n"
            b'event: error\r\ndata: {"error": "boom"}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        upstream = _RecordingTransport(
            httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        )
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 502
        assert json.loads(response.content) == {"error": "boom"}

    async def test_translates_nested_error_object_to_http_error(self):
        """``{"error": {"message": "..."}}`` — the shape used by OpenAI's
        own API — must also be recognized."""
        body = (
            b'data: {"error": {"message": "upstream crashed", "type": "x"}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        upstream = _RecordingTransport(
            httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        )
        transport = SSEToJSONBridgeTransport(upstream=upstream)

        response = await transport.handle_async_request(
            _chat_completions_request({"model": "x", "messages": []})
        )

        assert response.status_code == 502
        assert json.loads(response.content) == {"error": "upstream crashed"}


@pytest.fixture
def rollout_context():
    from osmosis_ai.rollout.context import RolloutContext

    ctx = RolloutContext(
        chat_completions_url="http://controller:9",
        api_key="test-key",
        rollout_id="rollout-xyz",
    )
    with ctx:
        yield ctx


class TestOsmosisOpenAIAgentWiresBridge:
    def test_async_openai_receives_bridge_transport(self, rollout_context):
        """OsmosisOpenAIAgent should wire its AsyncOpenAI through the SSE
        bridge transport."""
        from unittest.mock import MagicMock, patch

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        captured = {}

        def fake_client(*, base_url, api_key, http_client=None, **kwargs):
            captured["http_client"] = http_client
            return MagicMock(name="AsyncOpenAI")

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI",
            side_effect=fake_client,
        ):
            OsmosisOpenAIAgent(name="main")

        assert isinstance(captured["http_client"], httpx.AsyncClient)
        transport = captured["http_client"]._transport
        assert isinstance(transport, SSEToJSONBridgeTransport)

    async def test_runner_consumes_chunk_backed_sse_response(self):
        """The bridge must adapt controller-style chunk SSE into a shape that
        openai-python/openai-agents can consume in non-streaming mode."""
        from agents import Agent, Runner
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI

        chunk = _chunk_payload(
            content="hi",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

        class _ChunkUpstream(httpx.AsyncBaseTransport):
            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                body = (f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n").encode()
                return httpx.Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                )

        client = AsyncOpenAI(
            base_url="http://controller.local",
            api_key="test-key",
            http_client=httpx.AsyncClient(
                transport=SSEToJSONBridgeTransport(upstream=_ChunkUpstream())
            ),
        )
        model = OpenAIChatCompletionsModel(
            model="osmosis-rollout",
            openai_client=client,
        )
        agent = Agent(name="main", model=model)

        result = await Runner.run(agent, "hello")

        assert result.final_output == "hi"
