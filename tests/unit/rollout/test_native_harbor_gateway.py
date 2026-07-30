"""Local wire round trips for the native Harbor translation gateway."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.backend.native_harbor.gateway import (
    NativeHarborTranslationGateway,
)
from osmosis_ai.rollout.server.app import create_rollout_server
from osmosis_ai.rollout.server.native_harbor_gateway import (
    install_native_harbor_gateway_routes,
)


@dataclass(frozen=True)
class _QueuedResponse:
    content_type: str
    body: bytes


@pytest.fixture
def fake_chat_upstream() -> Any:
    responses: queue.Queue[_QueuedResponse] = queue.Queue()
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length))
            requests.append(
                {
                    "path": self.path,
                    "headers": {
                        key.lower(): value for key, value in self.headers.items()
                    },
                    "body": body,
                }
            )
            queued = responses.get_nowait()
            self.send_response(200)
            self.send_header("content-type", queued.content_type)
            self.send_header("content-length", str(len(queued.body)))
            self.end_headers()
            self.wfile.write(queued.body)

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def enqueue_json(response: dict[str, Any]) -> None:
        responses.put(
            _QueuedResponse(
                content_type="application/json",
                body=json.dumps(response).encode(),
            )
        )

    def enqueue_sse(chunks: list[dict[str, Any]]) -> None:
        frames = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
        frames.append("data: [DONE]\n\n")
        responses.put(
            _QueuedResponse(
                content_type="text/event-stream",
                body="".join(frames).encode(),
            )
        )

    yield SimpleNamespace(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        enqueue_json=enqueue_json,
        enqueue_sse=enqueue_sse,
        requests=requests,
    )

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _chat_response(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/osmosis-rollout",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        },
    }


def _chat_text_stream() -> list[dict[str, Any]]:
    common = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "openai/osmosis-rollout",
    }
    return [
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **common,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            **common,
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        },
    ]


def _gateway_app(
    fake_chat_upstream: Any,
    *,
    controller_api_key: str | None,
) -> tuple[FastAPI, NativeHarborTranslationGateway, str]:
    gateway = NativeHarborTranslationGateway("http://gateway.example:8000")
    token = gateway.register(
        chat_completions_url=fake_chat_upstream.base_url,
        controller_api_key=controller_api_key,
    )
    app = FastAPI()
    install_native_harbor_gateway_routes(app, gateway)
    return app, gateway, token


def test_rollout_server_mounts_configured_gateway_routes() -> None:
    with pytest.warns(UserWarning, match="codex.*eval-only"):
        backend = NativeHarborBackend(
            agent_name="codex",
            gateway_base_url="http://gateway.example:8000",
        )

    app = create_rollout_server(backend=backend, configure_logging=False)

    paths = {route.path for route in app.routes}
    assert {"/v1/messages", "/v1/responses"}.issubset(paths)


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
async def test_expired_gateway_route_is_rejected_without_upstream_call(
    fake_chat_upstream: Any,
    path: str,
) -> None:
    app, gateway, token = _gateway_app(
        fake_chat_upstream,
        controller_api_key="controller-real",
    )
    gateway.unregister(token)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post(
            path,
            headers={"authorization": f"Bearer {token}"},
            json={},
        )

    assert response.status_code == 401
    assert "expired" in response.text
    assert fake_chat_upstream.requests == []


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {
            "authorization": "Bearer route-a",
            "x-api-key": "route-b",
        },
    ],
)
async def test_gateway_rejects_missing_or_conflicting_credentials(
    fake_chat_upstream: Any,
    headers: dict[str, str],
) -> None:
    app, _, _ = _gateway_app(
        fake_chat_upstream,
        controller_api_key="controller-real",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post("/v1/responses", headers=headers, json={})

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert fake_chat_upstream.requests == []


async def test_anthropic_messages_round_trip_replaces_auth_and_translates_text(
    fake_chat_upstream: Any,
) -> None:
    fake_chat_upstream.enqueue_json(_chat_response())
    app, _, token = _gateway_app(
        fake_chat_upstream,
        controller_api_key="controller-real",
    )
    payload = {
        "model": "openai/osmosis-rollout",
        "max_tokens": 50,
        "system": "Be concise",
        "messages": [{"role": "user", "content": "hello"}],
        "stop_sequences": ["END"],
        "top_k": 4,
        "tools": [
            {
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
    }
    payload["extra_headers"] = {
        "host": "evil.example",
        "x-api-key": token,
    }
    payload["headers"] = {"cookie": f"route={token}"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post(
            "/v1/messages",
            headers={"x-api-key": token},
            json=payload,
        )

    assert response.status_code == 200
    result = response.json()
    assert result["type"] == "message"
    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert result["stop_reason"] == "end_turn"
    upstream = fake_chat_upstream.requests[0]
    assert upstream["path"] == "/v1/chat/completions"
    assert upstream["headers"]["authorization"] == "Bearer controller-real"
    assert token not in json.dumps(upstream)
    assert upstream["headers"]["host"] != "evil.example"
    assert "extra_headers" not in upstream["body"]
    assert "headers" not in upstream["body"]
    assert upstream["body"]["stop"] == ["END"]
    assert "stop_sequences" not in upstream["body"]
    assert "top_k" not in upstream["body"]
    assert upstream["body"]["tools"][0]["function"]["name"] == "lookup"


async def test_openai_responses_round_trip_uses_no_key_sentinel_and_translates_tool(
    fake_chat_upstream: Any,
) -> None:
    fake_chat_upstream.enqueue_json(
        _chat_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        )
    )
    app, _, token = _gateway_app(fake_chat_upstream, controller_api_key=None)
    payload = {
        "model": "openai/osmosis-rollout",
        "input": "look up x",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
    }
    payload["extra_headers"] = {
        "host": "evil.example",
        "authorization": f"Bearer {token}",
    }
    payload["headers"] = {"cookie": f"route={token}"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={"authorization": f"Bearer {token}"},
            json=payload,
        )

    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "response"
    assert result["status"] == "completed"
    function_call = next(
        item for item in result["output"] if item["type"] == "function_call"
    )
    assert function_call["name"] == "lookup"
    assert function_call["arguments"] == '{"q":"x"}'
    upstream = fake_chat_upstream.requests[0]
    assert upstream["path"] == "/v1/chat/completions"
    assert upstream["headers"]["authorization"] == ("Bearer osmosis-no-controller-key")
    assert token not in json.dumps(upstream)
    assert upstream["headers"]["host"] != "evil.example"
    assert "extra_headers" not in upstream["body"]
    assert "headers" not in upstream["body"]


@pytest.mark.parametrize(
    ("path", "header", "payload", "expected_events"),
    [
        (
            "/v1/messages",
            "x-api-key",
            {
                "model": "openai/osmosis-rollout",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            ["message_start", "content_block_delta", "message_stop"],
        ),
        (
            "/v1/responses",
            "authorization",
            {
                "model": "openai/osmosis-rollout",
                "input": "hello",
                "stream": True,
            },
            ["response.created", "response.output_text.delta", "response.completed"],
        ),
    ],
)
async def test_gateway_streams_translated_protocol_events(
    fake_chat_upstream: Any,
    path: str,
    header: str,
    payload: dict[str, Any],
    expected_events: list[str],
) -> None:
    fake_chat_upstream.enqueue_sse(_chat_text_stream())
    app, _, token = _gateway_app(
        fake_chat_upstream,
        controller_api_key="controller-real",
    )
    header_value = f"Bearer {token}" if header == "authorization" else token

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post(
            path,
            headers={header: header_value},
            json=payload,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    for event in expected_events:
        assert event in response.text
    assert "hello" in response.text
    upstream = fake_chat_upstream.requests[0]
    assert upstream["body"]["stream"] is True
    assert upstream["body"]["stream_options"] == {"include_usage": True}


async def test_responses_stream_keeps_response_and_item_ids_stable(
    fake_chat_upstream: Any,
) -> None:
    fake_chat_upstream.enqueue_sse(_chat_text_stream())
    app, _, token = _gateway_app(
        fake_chat_upstream,
        controller_api_key="controller-real",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.example:8000",
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={"authorization": f"Bearer {token}"},
            json={
                "model": "openai/osmosis-rollout",
                "input": "hello",
                "stream": True,
            },
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    created = next(event for event in events if event["type"] == "response.created")
    added = next(
        event for event in events if event["type"] == "response.output_item.added"
    )
    delta = next(
        event for event in events if event["type"] == "response.output_text.delta"
    )
    item_done = next(
        event for event in events if event["type"] == "response.output_item.done"
    )
    completed = next(event for event in events if event["type"] == "response.completed")

    response_id = created["response"]["id"]
    item_id = added["item"]["id"]
    assert completed["response"]["id"] == response_id
    assert delta["item_id"] == item_id
    assert item_done["item"]["id"] == item_id
    assert completed["response"]["output"][0]["id"] == item_id
