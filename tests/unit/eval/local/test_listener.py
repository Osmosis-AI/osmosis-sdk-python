from __future__ import annotations

import socket

import pytest

from osmosis_ai.eval.local.listener import LlmBridgeListener, create_llm_bridge_app
from osmosis_ai.eval.local.llm_bridge import LiteLLMBridge


def test_bridge_app_disables_schema_routes() -> None:
    app = create_llm_bridge_app(LiteLLMBridge(model="test/model"), auth_token="token")
    paths = {route.path for route in app.routes}
    assert "/docs" not in paths
    assert "/openapi.json" not in paths


async def test_listener_uses_a_reserved_loopback_port() -> None:
    listener = LlmBridgeListener(LiteLLMBridge(model="test/model"), auth_token="token")
    await listener.start()
    try:
        assert listener.base_url.startswith("http://127.0.0.1:")
        assert listener.chat_completions_url("a/b").endswith("/v1/rollouts/a%2Fb")
    finally:
        await listener.stop()


async def test_listener_rejects_an_occupied_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    listener = LlmBridgeListener(
        LiteLLMBridge(model="test/model"), auth_token="token", port=port
    )
    try:
        with pytest.raises(OSError):
            await listener.start()
    finally:
        sock.close()


async def test_advertised_url_is_used_for_chat_requests() -> None:
    listener = LlmBridgeListener(
        LiteLLMBridge(model="test/model"),
        auth_token="token",
        advertised_base_url="https://example.test/base/",
    )
    assert listener.chat_completions_url("r1") == (
        "https://example.test/base/v1/rollouts/r1"
    )
