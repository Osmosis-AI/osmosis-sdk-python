"""OpenAI endpoint selection must agree with actual LiteLLM request routing."""

from __future__ import annotations

import json

import httpx
import pytest

from osmosis_ai.eval.local.llm_bridge import LiteLLMBridge
from osmosis_ai.eval.request_policy import build_chat_kwargs

OFFICIAL_BASE = "https://api.openai.com/v1"
CUSTOM_BASE = "https://compatible.example/v1"


@pytest.mark.parametrize(
    ("explicit", "global_base", "base_url_env", "api_base_env", "expected"),
    [
        (None, None, None, None, OFFICIAL_BASE),
        (OFFICIAL_BASE, None, None, None, OFFICIAL_BASE),
        (None, OFFICIAL_BASE, None, None, OFFICIAL_BASE),
        (None, None, OFFICIAL_BASE, None, OFFICIAL_BASE),
        (None, None, None, OFFICIAL_BASE, OFFICIAL_BASE),
        (CUSTOM_BASE, None, None, None, CUSTOM_BASE),
        (None, CUSTOM_BASE, None, None, CUSTOM_BASE),
        (None, None, CUSTOM_BASE, None, CUSTOM_BASE),
        (None, None, None, CUSTOM_BASE, CUSTOM_BASE),
        (OFFICIAL_BASE, CUSTOM_BASE, CUSTOM_BASE, CUSTOM_BASE, OFFICIAL_BASE),
        (CUSTOM_BASE, OFFICIAL_BASE, OFFICIAL_BASE, OFFICIAL_BASE, CUSTOM_BASE),
        (None, OFFICIAL_BASE, CUSTOM_BASE, CUSTOM_BASE, OFFICIAL_BASE),
        (None, None, OFFICIAL_BASE, CUSTOM_BASE, OFFICIAL_BASE),
        (OFFICIAL_BASE + "/", None, None, None, OFFICIAL_BASE),
    ],
)
async def test_endpoint_and_inference_parameters_use_litellm_precedence(
    monkeypatch, explicit, global_base, base_url_env, api_base_env, expected
):
    import litellm

    monkeypatch.setattr(litellm, "api_base", global_base)
    for name, value in (
        ("OPENAI_BASE_URL", base_url_env),
        ("OPENAI_API_BASE", api_base_env),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    requests = []

    async def send(_client, request, **_kwargs):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-only"
        if request.url.path.endswith("/responses"):
            payload = {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "osmosis-rollout",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        else:
            payload = {
                "id": "chat_test",
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
            }
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "low",
        "custom_option": {"enabled": True},
    }
    kwargs = build_chat_kwargs(body, model="openai/osmosis-rollout", api_base=explicit)
    official = expected == OFFICIAL_BASE
    if official:
        assert kwargs["reasoning_effort"] == "low"
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"]["reasoning_effort"] == "low"
        assert kwargs["extra_body"]["custom_option"] == {"enabled": True}

    bridge = LiteLLMBridge(
        model="openai/osmosis-rollout", api_key="test-only", api_base=explicit
    )
    await bridge.complete(body, rollout_id="endpoint-selection")

    (request,) = requests
    endpoint = "/responses" if official else "/chat/completions"
    assert str(request.url) == expected + endpoint
    payload = json.loads(request.content)
    assert payload["custom_option"] == {"enabled": True}
    if official:
        assert payload["reasoning"] == {"effort": "low"}
    else:
        assert payload["reasoning_effort"] == "low"
