"""The contract stub's real-provider passthrough.

The stub's canned reply is enough to lock the wire contract but scores zero
against any real grader, so a local end-to-end run points it at a provider. The
passthrough must keep the frozen request/SSE contract and, critically, surface an
upstream error as an error rather than a truncated stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from tests.unit.rollout.eval_proxy_stub import (
    DEFAULT_STUB_PLATFORM_TOKEN,
    EvalProxyStubUpstream,
    create_eval_proxy_stub_app,
)

PLATFORM_TOKEN = DEFAULT_STUB_PLATFORM_TOKEN
ROLLOUT_ID = "a" * 32

_SSE_BODY = (
    'data: {"id":"c1","object":"chat.completion.chunk","choices":'
    '[{"index":0,"delta":{"role":"assistant","content":"42"},"finish_reason":null}]}\n'
    "\n"
    'data: {"id":"c1","object":"chat.completion.chunk","choices":'
    '[{"index":0,"delta":{},"finish_reason":"stop"}]}\n'
    "\n"
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
    '"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)


class _Upstream:
    """A fake provider that records what the stub forwarded to it."""

    def __init__(self, *, status: int = 200, body: str = _SSE_BODY) -> None:
        self.status = status
        self.body = body
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.status >= 400:
            return httpx.Response(
                self.status, json={"error": {"message": "unsupported parameter"}}
            )
        return httpx.Response(
            200, text=self.body, headers={"content-type": "text/event-stream"}
        )


@pytest.fixture
def upstream_client(monkeypatch: pytest.MonkeyPatch) -> _Upstream:
    """Route the stub's outbound provider calls to an in-process fake."""
    fake = _Upstream()
    real_client = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(fake.handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr("tests.unit.rollout.eval_proxy_stub.httpx.AsyncClient", patched)
    return fake


@asynccontextmanager
async def _session_client(app: Any) -> AsyncIterator[tuple[httpx.AsyncClient, str]]:
    """Open a client against the stub app and create one session on it."""
    from httpx import ASGITransport

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://stub"
    ) as client:
        response = await client.post(
            "/v1/eval-sessions",
            json={"rollout_id": ROLLOUT_ID, "model_path": "openai/gpt-4o-mini"},
            headers={"Authorization": f"Bearer {PLATFORM_TOKEN}"},
        )
        assert response.status_code == 200
        yield client, response.json()["token"]


def _app(upstream_base: str = "https://provider.test/v1") -> Any:
    return create_eval_proxy_stub_app(
        platform_token=PLATFORM_TOKEN,
        upstream=EvalProxyStubUpstream(base_url=upstream_base, api_key="sk-fake"),
    )


async def test_the_wire_model_is_mapped_to_the_session_model_path(
    upstream_client: _Upstream,
) -> None:
    app = _app()
    async with _session_client(app) as (client, token):
        response = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={
                "model": "osmosis-rollout",
                "messages": [{"role": "user", "content": "2*21"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
    forwarded = upstream_client.requests[0]
    # The provider prefix is stripped, exactly as the locked LiteLLM path does.
    assert forwarded["model"] == "gpt-4o-mini"
    assert forwarded["stream"] is True
    # The relayed stream must still end with the contract's usage-only chunk,
    # whether or not the client asked for one.
    assert forwarded["stream_options"] == {"include_usage": True}


async def test_the_upstream_stream_is_relayed_verbatim(
    upstream_client: _Upstream,
) -> None:
    app = _app()
    async with _session_client(app) as (client, token):
        response = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={
                "model": "osmosis-rollout",
                "messages": [{"role": "user", "content": "2*21"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert "42" in response.text
        assert response.text.rstrip().endswith("data: [DONE]")


async def test_an_upstream_error_surfaces_as_a_status_not_a_truncated_stream(
    upstream_client: _Upstream,
) -> None:
    # Raising inside the SSE generator would send 200 + headers first, so the
    # client would report a transport error instead of the provider's message.
    upstream_client.status = 400
    app = _app()
    async with _session_client(app) as (client, token):
        response = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={
                "model": "osmosis-rollout",
                "messages": [{"role": "user", "content": "2*21"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "upstream provider returned 400" in detail
    assert "unsupported parameter" in detail


async def test_the_frozen_request_contract_still_applies_with_a_passthrough(
    upstream_client: _Upstream,
) -> None:
    app = _app()
    async with _session_client(app) as (client, token):
        non_streaming = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={"model": "osmosis-rollout", "messages": [], "stream": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        wrong_model = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [], "stream": True},
            headers={"Authorization": f"Bearer {token}"},
        )
        unauthorized = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={"model": "osmosis-rollout", "messages": [], "stream": True},
            headers={"Authorization": "Bearer wrong"},
        )
    assert non_streaming.status_code == 400
    assert wrong_model.status_code == 400
    assert unauthorized.status_code == 401
    assert upstream_client.requests == []


async def test_the_canned_stub_is_unchanged_without_a_passthrough() -> None:
    app = create_eval_proxy_stub_app(platform_token=PLATFORM_TOKEN)
    async with _session_client(app) as (client, token):
        response = await client.post(
            f"/v1/eval-sessions/{ROLLOUT_ID}/chat/completions",
            json={"model": "osmosis-rollout", "messages": [], "stream": True},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert "chatcmpl-stub" in response.text
