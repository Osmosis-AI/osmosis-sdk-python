"""Tests for the in-process LiteLLM bridge."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from osmosis_ai.rollout.controller import llm_bridge
from osmosis_ai.rollout.controller.llm_bridge import (
    LiteLLMBridge,
    create_bridge_router,
)

ROLLOUT_ID = "a" * 32
BRIDGE_TOKEN = "bridge-secret"


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class InternalServerError(Exception):
    pass


class BadRequestError(Exception):
    def __init__(
        self,
        *,
        param: str,
        message: str = "Unsupported parameter",
        error_type: str = "invalid_request_error",
        lossy: bool = False,
    ) -> None:
        details = {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
        rendered_message = (
            f"litellm.BadRequestError: OpenAIException - "
            f"{json.dumps({'error': details})}"
            if lossy
            else message
        )
        super().__init__(rendered_message)
        self.message = rendered_message
        self.status_code = 400
        self.llm_provider = "openai"
        self.type = None if lossy else error_type
        self.param = None if lossy else param
        self.body = None if lossy else details


def _response(
    content: str = "hello",
    *,
    total_tokens: int = 5,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "created": 1700000000,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": total_tokens,
        },
    }


def _responses_response(
    *,
    output: list[dict[str, Any]] | None = None,
    total_tokens: int = 5,
    status: str = "completed",
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "id": "resp-test",
        "created_at": 1700000000,
        "status": status,
        "output": output
        if output is not None
        else [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": total_tokens,
        },
    }
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    if error is not None:
        response["error"] = error
    return response


class _FakeLiteLLM:
    """Duck-typed litellm module: records calls, returns or raises."""

    BadRequestError = BadRequestError

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        responses_response: dict[str, Any] | None = None,
        completion_error: Exception | None = None,
        responses_error: Exception | None = None,
        responses_errors: list[Exception] | None = None,
        provider_error: Exception | None = None,
        unsupported_openai_params: set[str] | None = None,
    ) -> None:
        self.suppress_debug_info = False
        self._response = response if response is not None else _response()
        self._responses_response = (
            responses_response
            if responses_response is not None
            else _responses_response()
        )
        self._completion_error = completion_error
        self._responses_error = responses_error
        self._responses_errors = list(responses_errors or [])
        self._provider_error = provider_error
        self._unsupported_openai_params = unsupported_openai_params or set()
        self.completion_kwargs: list[dict[str, Any]] = []
        self.responses_kwargs: list[dict[str, Any]] = []
        self.optional_param_calls: list[dict[str, Any]] = []
        self.provider_calls: list[dict[str, Any]] = []

    def get_llm_provider(self, *, model: str, api_base: str | None = None) -> None:
        self.provider_calls.append({"model": model, "api_base": api_base})
        if self._provider_error is not None:
            raise self._provider_error

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        self.completion_kwargs.append(kwargs)
        if self._completion_error is not None:
            raise self._completion_error
        return self._response

    async def aresponses(self, **kwargs: Any) -> dict[str, Any]:
        self.responses_kwargs.append(kwargs)
        if self._responses_errors:
            raise self._responses_errors.pop(0)
        if self._responses_error is not None:
            raise self._responses_error
        return self._responses_response

    def get_optional_params(self, **kwargs: Any) -> dict[str, Any]:
        self.optional_param_calls.append(kwargs)
        return {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "model",
                "custom_llm_provider",
                "drop_params",
                *self._unsupported_openai_params,
            }
        }


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeLiteLLM) -> None:
    monkeypatch.setattr(llm_bridge, "_get_litellm", lambda: fake)


def _client(bridge: LiteLLMBridge) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(create_bridge_router(bridge, auth_token=BRIDGE_TOKEN))
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://127.0.0.1",
    )


def _url(rollout_id: str = ROLLOUT_ID) -> str:
    return f"/v1/rollouts/{rollout_id}/chat/completions"


def _auth(token: str = BRIDGE_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _body(**extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "hi"}], **extra}


async def test_missing_or_wrong_bearer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM())
    async with _client(LiteLLMBridge(model="anthropic/claude-test")) as client:
        assert (await client.post(_url(), json=_body())).status_code == 401
        assert (
            await client.post(_url(), json=_body(), headers=_auth("wrong"))
        ).status_code == 401


async def test_non_stream_returns_openai_payload_and_accounts_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 5
    assert bridge.collect_tokens(ROLLOUT_ID) == 5
    # collect_tokens pops: a second read reports "never served".
    assert bridge.collect_tokens(ROLLOUT_ID) is None


async def test_tokens_accumulate_across_calls_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        await client.post(_url(), json=_body(), headers=_auth())
        await client.post(_url(), json=_body(), headers=_auth())
    assert bridge.collect_tokens(ROLLOUT_ID) == 10
    assert bridge.collect_tokens("b" * 32) is None


async def test_stream_serves_single_chunk_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM())
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        resp = await client.post(_url(), json=_body(stream=True), headers=_auth())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert ": ping" in text
    assert '"object":"chat.completion.chunk"' in text
    assert '"content":"hello"' in text
    assert '"finish_reason":"stop"' in text
    assert '"total_tokens":5' in text
    assert text.rstrip().endswith("data: [DONE]")


async def test_stream_error_emits_error_event_then_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(completion_error=InternalServerError("boom")))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        resp = await client.post(_url(), json=_body(stream=True), headers=_auth())
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")


async def test_non_stream_error_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(completion_error=InternalServerError("boom")))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 502
    assert bridge.collect_tokens(ROLLOUT_ID) is None


async def test_empty_choices_fail_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(response={"id": "x", "choices": []}))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 502
    assert "no choices" in resp.json()["detail"]


async def test_invalid_rollout_id_and_body_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM())
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        # %5C decodes to a backslash: it routes as one segment but is not a
        # portable single path component, so the handler rejects it.
        nested = await client.post(
            "/v1/rollouts/a%5Cb/chat/completions", json=_body(), headers=_auth()
        )
        assert nested.status_code == 422
        bad_body = await client.post(_url(), json=["not", "a", "dict"], headers=_auth())
        assert bad_body.status_code == 400


async def test_request_fields_are_filtered_and_credentials_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(
        model="anthropic/claude-test",
        api_key="provider-key",
        api_base="http://127.0.0.1:9/v1",
    )
    async with _client(bridge) as client:
        await client.post(
            _url(),
            json=_body(
                model="whatever-the-client-said",
                temperature=0.5,
                max_tokens=32,
                stream_options={"include_usage": True},
                unknown_field="dropped",
            ),
            headers=_auth(),
        )
    (kwargs,) = fake.completion_kwargs
    assert kwargs["model"] == "anthropic/claude-test"
    assert kwargs["api_key"] == "provider-key"
    assert kwargs["base_url"] == "http://127.0.0.1:9/v1"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 32
    assert kwargs["drop_params"] is True
    assert "stream_options" not in kwargs
    assert "unknown_field" not in kwargs
    assert "stream" not in kwargs


async def test_tool_calls_survive_both_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
    ]
    _install(monkeypatch, _FakeLiteLLM(response=_response(tool_calls=tool_calls)))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        plain = await client.post(_url(), json=_body(), headers=_auth())
        streamed = await client.post(_url(), json=_body(stream=True), headers=_auth())
    assert plain.json()["choices"][0]["message"]["tool_calls"] == tool_calls
    # The stream delta shape adds the per-item index OpenAI clients expect.
    assert '"tool_calls":[{"index":0,' in streamed.text


async def test_official_openai_uses_responses_api_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    function_call = {
        "type": "function_call",
        "call_id": "call_2",
        "name": "multiply",
        "arguments": '{"a":6,"b":7}',
    }
    fake = _FakeLiteLLM(
        responses_response=_responses_response(
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Using the tool."}],
                },
                function_call,
            ]
        )
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test", api_key="provider-key")
    prior_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "multiply", "arguments": '{"a":2,"b":3}'},
    }
    tool = {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "What is 2 * 3?"},
        {"role": "assistant", "content": "", "tool_calls": [prior_tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "6"},
        {"role": "user", "content": "Now multiply 6 * 7."},
    ]

    async with _client(bridge) as client:
        resp = await client.post(
            _url(),
            json=_body(
                messages=messages,
                tools=[tool],
                temperature=0.5,
                top_p=0.9,
                max_tokens=128,
            ),
            headers=_auth(),
        )

    assert resp.status_code == 200
    assert fake.completion_kwargs == []
    (kwargs,) = fake.responses_kwargs
    assert kwargs["model"] == "openai/gpt-test"
    assert kwargs["api_key"] == "provider-key"
    assert kwargs["max_output_tokens"] == 128
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["tools"] == [{"type": "function", **tool["function"]}]
    assert {
        "type": "function_call",
        "call_id": "call_1",
        **prior_tool_call["function"],
    } in (kwargs["input"])
    assert {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "6",
    } in kwargs["input"]

    payload = resp.json()
    choice = payload["choices"][0]
    assert choice["message"]["content"] == "Using the tool."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
        }
    ]
    assert choice["finish_reason"] == "tool_calls"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert bridge.collect_tokens(ROLLOUT_ID) == 5


async def test_official_openai_drops_unsupported_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(unsupported_openai_params={"top_p"})
    _install(monkeypatch, fake)

    bridge = LiteLLMBridge(model="openai/gpt-5-mini")
    async with _client(bridge) as client:
        resp = await client.post(
            _url(),
            json=_body(temperature=1.0, top_p=0.9),
            headers=_auth(),
        )

    assert resp.status_code == 200
    assert fake.optional_param_calls == [
        {
            "model": "openai/gpt-5-mini",
            "custom_llm_provider": "openai",
            "drop_params": True,
            "temperature": 1.0,
            "top_p": 0.9,
        }
    ]
    (kwargs,) = fake.responses_kwargs
    assert kwargs["temperature"] == 1.0
    assert "top_p" not in kwargs


async def test_official_openai_recovers_from_stale_sampling_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(
        responses_errors=[
            BadRequestError(
                param="top_p",
                message=(
                    "Unsupported parameter: 'top_p' is not supported with this model."
                ),
                lossy=True,
            )
        ]
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-5.6-luna")

    async with _client(bridge) as client:
        recovered = await client.post(
            _url(),
            json=_body(temperature=1.0, top_p=0.9),
            headers=_auth(),
        )
        cached = await client.post(
            _url(),
            json=_body(temperature=1.0, top_p=0.9),
            headers=_auth(),
        )

    assert recovered.status_code == 200
    assert cached.status_code == 200
    first, retry, later = fake.responses_kwargs
    assert first["top_p"] == 0.9
    assert "top_p" not in retry
    assert "top_p" not in later
    assert retry["temperature"] == later["temperature"] == 1.0


async def test_official_openai_does_not_retry_other_bad_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(
        responses_error=BadRequestError(
            param="top_p",
            message="Invalid value for 'top_p': expected a number between 0 and 1.",
        )
    )
    _install(monkeypatch, fake)

    async with _client(LiteLLMBridge(model="openai/gpt-test")) as client:
        response = await client.post(
            _url(),
            json=_body(top_p=2.0),
            headers=_auth(),
        )

    assert response.status_code == 502
    assert len(fake.responses_kwargs) == 1


async def test_official_openai_converts_chat_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at these."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8=",
                        "detail": "low",
                    },
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:text/plain;base64,aGVsbG8=",
                        "filename": "hello.txt",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Calling the tool."}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "done"}],
        },
    ]

    await bridge.complete({"messages": messages}, rollout_id=ROLLOUT_ID)

    (kwargs,) = fake.responses_kwargs
    assert kwargs["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at these."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aGVsbG8=",
                    "detail": "low",
                },
                {
                    "type": "input_file",
                    "file_data": "data:text/plain;base64,aGVsbG8=",
                    "filename": "hello.txt",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Calling the tool."}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "done"}],
        },
    ]


async def test_official_openai_replays_a_prior_refusal_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")
    messages = [
        {"role": "user", "content": "Do the thing."},
        {"role": "assistant", "content": None, "refusal": "I cannot help with that."},
        {"role": "user", "content": "Then do a safer thing."},
    ]

    await bridge.complete({"messages": messages}, rollout_id=ROLLOUT_ID)

    (kwargs,) = fake.responses_kwargs
    assert kwargs["input"][1] == {
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }


async def test_official_openai_forwards_modern_chat_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")

    await bridge.complete(
        _body(
            max_completion_tokens=64,
            max_tokens=128,
            tool_choice={"type": "function", "function": {"name": "multiply"}},
            parallel_tool_calls=False,
        ),
        rollout_id=ROLLOUT_ID,
    )

    (kwargs,) = fake.responses_kwargs
    assert kwargs["max_output_tokens"] == 64
    assert kwargs["tool_choice"] == {"type": "function", "name": "multiply"}
    assert kwargs["parallel_tool_calls"] is False


@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
    ],
)
async def test_official_openai_maps_incomplete_status(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    finish_reason: str,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(
        responses_response=_responses_response(
            status="incomplete",
            incomplete_reason=reason,
        )
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")

    async with _client(bridge) as client:
        response = await client.post(_url(), json=_body(), headers=_auth())

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    ("status", "incomplete_reason", "error", "expected_detail"),
    [
        (
            "failed",
            None,
            {"code": "invalid_prompt", "message": "Prompt rejected."},
            "invalid_prompt: Prompt rejected.",
        ),
        ("cancelled", None, None, "unsuccessful status 'cancelled'"),
        ("in_progress", None, None, "unsuccessful status 'in_progress'"),
        ("queued", None, None, "unsuccessful status 'queued'"),
        ("incomplete", "future_reason", None, "unknown reason 'future_reason'"),
    ],
)
async def test_official_openai_rejects_unsuccessful_status(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    incomplete_reason: str | None,
    error: dict[str, Any] | None,
    expected_detail: str,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(
        responses_response=_responses_response(
            status=status,
            incomplete_reason=incomplete_reason,
            error=error,
        )
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")

    async with _client(bridge) as client:
        response = await client.post(_url(), json=_body(), headers=_auth())

    assert response.status_code == 502
    assert expected_detail in response.json()["detail"]


async def test_official_openai_preserves_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM(
        responses_response=_responses_response(
            output=[
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "I cannot help with that."}
                    ],
                }
            ]
        )
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test")

    async with _client(bridge) as client:
        response = await client.post(_url(), json=_body(), headers=_auth())
        streamed = await client.post(_url(), json=_body(stream=True), headers=_auth())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "",
        "refusal": "I cannot help with that.",
    }
    assert '"refusal":"I cannot help with that."' in streamed.text


@pytest.mark.parametrize(
    ("api_base", "env_name"),
    [
        ("https://compatible.example/v1", None),
        (None, "OPENAI_BASE_URL"),
        (None, "OPENAI_API_BASE"),
    ],
)
async def test_custom_openai_base_keeps_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
    api_base: str | None,
    env_name: str | None,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    if env_name:
        monkeypatch.setenv(env_name, "https://compatible.example/v1")
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openai/gpt-test", api_base=api_base)

    await bridge.complete(_body(), rollout_id=ROLLOUT_ID)

    assert len(fake.completion_kwargs) == 1
    assert fake.responses_kwargs == []


async def test_openrouter_openai_model_keeps_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="openrouter/openai/gpt-test")

    await bridge.complete(_body(), rollout_id=ROLLOUT_ID)

    assert len(fake.completion_kwargs) == 1
    assert fake.responses_kwargs == []


async def test_preflight_rejects_unknown_model_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(provider_error=ValueError("unknown provider")))
    bridge = LiteLLMBridge(model="not-a-provider-model")
    with pytest.raises(RuntimeError, match="Invalid LiteLLM model format"):
        await bridge.preflight_check()


async def test_preflight_raises_on_fatal_and_tolerates_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(completion_error=AuthenticationError("bad key")))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    with pytest.raises(AuthenticationError):
        await bridge.preflight_check()

    _install(monkeypatch, _FakeLiteLLM(completion_error=RateLimitError("slow down")))
    await bridge.preflight_check()

    _install(monkeypatch, _FakeLiteLLM(completion_error=InternalServerError("flaky")))
    await bridge.preflight_check()


async def test_preflight_probes_official_openai_through_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    await LiteLLMBridge(model="openai/gpt-test").preflight_check()
    (kwargs,) = fake.responses_kwargs
    assert fake.completion_kwargs == []
    assert kwargs["max_output_tokens"] == 256

    # The preflight error contract holds on the Responses path too: fatal
    # config errors raise, rate limits and flaky upstreams are tolerated.
    _install(monkeypatch, _FakeLiteLLM(responses_error=AuthenticationError("bad key")))
    with pytest.raises(AuthenticationError):
        await LiteLLMBridge(model="openai/gpt-test").preflight_check()

    _install(monkeypatch, _FakeLiteLLM(responses_error=RateLimitError("slow down")))
    await LiteLLMBridge(model="openai/gpt-test").preflight_check()

    _install(monkeypatch, _FakeLiteLLM(responses_error=InternalServerError("flaky")))
    await LiteLLMBridge(model="openai/gpt-test").preflight_check()


async def test_bridge_router_requires_non_empty_token() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        create_bridge_router(
            LiteLLMBridge(model="anthropic/claude-test"), auth_token="  "
        )
