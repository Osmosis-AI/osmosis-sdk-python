"""Parameter translation and actual LiteLLM Responses request serialization."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

from osmosis_ai.eval.local.openai_responses import build_responses_kwargs
from osmosis_ai.eval.request_policy import BridgeRequestError


def _kwargs(body: dict[str, Any]) -> dict[str, Any]:
    return build_responses_kwargs(body, model="openai/gpt-4.1", api_key="test-only")


def test_merges_custom_parameters_without_mutating_the_request() -> None:
    body = {
        "model": "rollout-model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "low"},
        "temperature": 0.2,
        "custom_option": {"enabled": False},
        "extra_body": {
            "reasoning": {"effort": "high", "summary": "auto"},
            "temperature": 0.7,
            "provider": {"order": ["example"]},
        },
    }
    original = deepcopy(body)

    kwargs = _kwargs(body)

    assert body == original
    assert kwargs["model"] == "openai/gpt-4.1"
    assert kwargs["input"] == body["messages"]
    assert kwargs["temperature"] == 0.7
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "high", "summary": "auto"},
        "custom_option": {"enabled": False},
        "provider": {"order": ["example"]},
    }


def test_translates_reasoning_and_structured_output_with_native_options() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    kwargs = _kwargs(
        {
            "reasoning_effort": "high",
            "reasoning": {"summary": "auto"},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            "text": {"verbosity": "low"},
        }
    )

    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "high", "summary": "auto"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": schema,
                "strict": True,
            },
        },
    }


@pytest.mark.parametrize("response_format", [{"type": "json_object"}, {"type": "text"}])
def test_translates_simple_response_formats(response_format: dict[str, Any]) -> None:
    assert _kwargs({"response_format": response_format})["extra_body"] == {
        "text": {"format": response_format}
    }


@pytest.mark.parametrize(
    "body",
    [
        {"reasoning_effort": "low", "reasoning": {"effort": "high"}},
        {"reasoning_effort": "low", "reasoning": "invalid"},
        {
            "response_format": {"type": "json_object"},
            "text": {"format": {"type": "text"}},
        },
    ],
)
def test_conflicting_protocol_aliases_raise_explicit_errors(
    body: dict[str, Any],
) -> None:
    with pytest.raises(BridgeRequestError, match="conflicting"):
        _kwargs(body)


def test_translates_tool_options_and_uses_native_token_limit() -> None:
    function = {"name": "lookup", "parameters": {"type": "object"}, "strict": True}
    kwargs = _kwargs(
        {
            "max_tokens": 256,
            "max_completion_tokens": 128,
            "extra_body": {"max_output_tokens": 64},
            "tools": [{"type": "function", "function": function}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "parallel_tool_calls": False,
        }
    )

    assert kwargs["max_output_tokens"] == 64
    assert kwargs["tools"] == [{"type": "function", **function}]
    assert kwargs["tool_choice"] == {"type": "function", "name": "lookup"}
    assert kwargs["parallel_tool_calls"] is False
    assert "extra_body" not in kwargs


def test_invalid_tools_are_not_silently_discarded() -> None:
    assert _kwargs({"tools": "invalid"})["tools"] == "invalid"


async def test_custom_and_translated_parameters_reach_litellm_http_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-4.1",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            AsyncHTTPHandler, "create_client", lambda self, **kw: client
        )
        await litellm.aresponses(
            **_kwargs(
                {
                    "model": "ignored-rollout-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "reasoning_effort": "low",
                    "max_completion_tokens": 64,
                    "response_format": {"type": "json_object"},
                    "custom_option": {"enabled": True},
                    "extra_body": {
                        "custom_option": {"enabled": False},
                        "provider": {"order": ["example"]},
                    },
                }
            ),
            client=AsyncHTTPHandler(),
        )

    (request,) = requests
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer test-only"
    assert json.loads(request.content) == {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 64,
        "text": {"format": {"type": "json_object"}},
        "custom_option": {"enabled": False},
        "provider": {"order": ["example"]},
    }


async def test_upstream_rejection_of_custom_parameter_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unknown parameter: 'custom_option'.",
                    "type": "invalid_request_error",
                    "param": "custom_option",
                    "code": "unknown_parameter",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            AsyncHTTPHandler, "create_client", lambda self, **kw: client
        )
        with pytest.raises(litellm.BadRequestError, match="custom_option"):
            await litellm.aresponses(
                **_kwargs({"custom_option": True}), client=AsyncHTTPHandler()
            )

    assert len(bodies) == 1
    assert bodies[0]["custom_option"] is True
