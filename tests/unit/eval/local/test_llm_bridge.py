"""Tests for the in-process LiteLLM bridge."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from osmosis_ai.eval.local import llm_bridge
from osmosis_ai.eval.local.llm_bridge import (
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


class _SlowLiteLLM(_FakeLiteLLM):
    """Completes (or fails) only after a delay, to outlive the grace window."""

    def __init__(self, *, delay: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay = delay

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(self._delay)
        return await super().acompletion(**kwargs)


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeLiteLLM) -> None:
    monkeypatch.setattr(llm_bridge, "_get_litellm", lambda: fake)


def _client(
    bridge: LiteLLMBridge, *, non_stream_keepalive: bool = False
) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(
        create_bridge_router(
            bridge,
            auth_token=BRIDGE_TOKEN,
            non_stream_keepalive=non_stream_keepalive,
        )
    )
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


@pytest.mark.parametrize("stream", [False, True])
async def test_missing_litellm_returns_optional_dependency_install_hint(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    # Exercise _get_litellm and the HTTP error handler without uninstalling extras.
    monkeypatch.setitem(sys.modules, "litellm", None)
    monkeypatch.setitem(
        sys.modules, "litellm.litellm_core_utils.secret_redaction", None
    )
    async with _client(LiteLLMBridge(model="anthropic/claude-test")) as client:
        response = await client.post(_url(), json=_body(stream=stream), headers=_auth())
    assert response.status_code == (200 if stream else 502)
    if stream:
        payload = next(
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        )
        assert response.text.rstrip().endswith("data: [DONE]")
    else:
        payload = response.json()
    assert 'pip install "osmosis-ai[eval]"' in payload["error"]["message"]
    assert payload["error"]["code"] == 502


def test_error_redacts_known_secrets_when_litellm_redactor_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "litellm.litellm_core_utils.secret_redaction", None
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-provider-test-secret")
    _, payload = llm_bridge._bridge_error(
        RuntimeError(
            "Provider failed: configured-secret, ambient-provider-test-secret"
        ),
        secret_values=("configured-secret",),
    )
    assert payload["error"]["message"] == "Provider failed: [REDACTED], [REDACTED]"


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


async def test_request_fields_are_forwarded_and_credentials_injected(
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
        response = await client.post(
            _url(),
            json=_body(
                model="whatever-the-client-said",
                temperature=0.5,
                max_tokens=32,
                stream=True,
                stream_options={"include_usage": True},
                unknown_field="forwarded",
                reasoning={"effort": "low"},
                extra_body={"provider": {"order": ["test-provider"]}},
            ),
            headers=_auth(),
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    (kwargs,) = fake.completion_kwargs
    assert kwargs["model"] == "anthropic/claude-test"
    assert kwargs["api_key"] == "provider-key"
    assert kwargs["base_url"] == "http://127.0.0.1:9/v1"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 32
    assert kwargs["drop_params"] is False
    assert "stream_options" not in kwargs
    assert kwargs["unknown_field"] == "forwarded"
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["provider"] == {"order": ["test-provider"]}
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
    chunk = next(
        json.loads(line[6:])
        for line in streamed.text.splitlines()
        if line.startswith("data: {")
    )
    assert chunk["choices"][0]["delta"]["tool_calls"] == [{**tool_calls[0], "index": 0}]


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


async def test_official_openai_does_not_filter_sampling_params(
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
    assert fake.optional_param_calls == []
    (kwargs,) = fake.responses_kwargs
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.9


async def test_official_openai_preserves_rejected_sampling_params_on_later_calls(
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
        rejected = await client.post(
            _url(),
            json=_body(temperature=1.0, top_p=0.9),
            headers=_auth(),
        )
        later = await client.post(
            _url(),
            json=_body(temperature=1.0, top_p=0.9),
            headers=_auth(),
        )

    assert rejected.status_code == 400
    assert "top_p" in rejected.json()["error"]["message"]
    assert later.status_code == 200
    first, second = fake.responses_kwargs
    assert first["top_p"] == 0.9
    assert second["top_p"] == 0.9
    assert first["temperature"] == second["temperature"] == 1.0


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

    assert response.status_code == 400
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


async def test_keepalive_fast_response_keeps_clean_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM())
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert bridge.collect_tokens(ROLLOUT_ID) == 5


async def test_keepalive_fast_error_keeps_clean_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeLiteLLM(completion_error=InternalServerError("boom")))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 502


async def test_keepalive_slow_response_trickles_whitespace_then_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.05)
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_KEEPALIVE_INTERVAL_SEC", 0.05)
    _install(monkeypatch, _SlowLiteLLM(delay=0.3))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    # The keepalive bytes are leading whitespace, so the body is still one
    # valid JSON document.
    assert resp.text.startswith(" ")
    payload = json.loads(resp.text)
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert bridge.collect_tokens(ROLLOUT_ID) == 5


async def test_keepalive_response_cleanup_cancels_an_unstarted_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request

    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.01)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_complete(body: dict[str, Any], *, rollout_id: str) -> Any:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(bridge, "complete", never_complete)
    router = create_bridge_router(
        bridge,
        auth_token=BRIDGE_TOKEN,
        non_stream_keepalive=True,
    )
    route = next(
        route for route in router.routes if route.path.endswith("/chat/completions")
    )
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": json.dumps(_body()).encode(),
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    request = Request(
        {"type": "http", "method": "POST", "path": _url(), "headers": []},
        receive,
    )
    response = await route.endpoint(ROLLOUT_ID, request)
    await started.wait()
    assert response.background is not None
    await response.background()
    assert cancelled.is_set()


async def test_keepalive_late_error_emits_openai_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.05)
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_KEEPALIVE_INTERVAL_SEC", 0.05)
    _install(
        monkeypatch,
        _SlowLiteLLM(delay=0.3, completion_error=InternalServerError("boom")),
    )
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    # The 200 was already committed; the error can only ride in the body.
    assert resp.status_code == 200
    payload = json.loads(resp.text)
    assert payload["error"]["type"] == "bridge_error"
    assert "boom" in payload["error"]["message"]


async def test_keepalive_provider_timeout_error_in_grace_is_clean_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TimeoutError == asyncio.TimeoutError since 3.11: a provider raising it
    # must never be mistaken for the grace timer expiring.
    _install(monkeypatch, _FakeLiteLLM(completion_error=TimeoutError("provider")))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 502


async def test_keepalive_provider_timeout_error_past_grace_terminates_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ...and past the grace window it must terminate the trickle with an
    # error body instead of spinning whitespace forever.
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.05)
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_KEEPALIVE_INTERVAL_SEC", 0.05)
    _install(
        monkeypatch,
        _SlowLiteLLM(delay=0.2, completion_error=TimeoutError("provider")),
    )
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    assert resp.status_code == 200
    payload = json.loads(resp.text)
    assert payload["error"]["type"] == "bridge_error"


async def test_keepalive_first_body_byte_is_immediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Committed headers do not reset proxy idle-read timers; the first body
    # byte must go out with the commit, not one keepalive interval later.
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.05)
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_KEEPALIVE_INTERVAL_SEC", 30.0)
    _install(monkeypatch, _SlowLiteLLM(delay=0.2))
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge, non_stream_keepalive=True) as client:
        resp = await client.post(_url(), json=_body(), headers=_auth())
    # One immediate whitespace byte, then the payload as soon as it is ready
    # (asyncio.wait wakes on completion, not on the 30s interval).
    assert resp.status_code == 200
    assert resp.text.startswith(" ")
    assert json.loads(resp.text)["choices"][0]["message"]["content"] == "hello"


async def test_bridge_router_requires_non_empty_token() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        create_bridge_router(
            LiteLLMBridge(model="anthropic/claude-test"), auth_token="  "
        )


@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_is_returned_and_replayed_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    tool_call = {
        "id": "call_reasoning",
        "type": "function",
        "index": 3,
        "function": {"name": "lookup", "arguments": "{}"},
    }
    reasoning = {
        "reasoning_content": "Use the lookup tool.",
        "thinking_blocks": [
            {"type": "thinking", "thinking": "Use the lookup tool.", "signature": "sig"}
        ],
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
    }
    response = _response(tool_calls=[tool_call])
    response["choices"][0]["message"].update(reasoning)
    fake = _FakeLiteLLM(response=response)
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    async with _client(bridge) as client:
        reply = await client.post(_url(), json=_body(stream=stream), headers=_auth())
    if stream:
        chunk = next(
            json.loads(line[6:])
            for line in reply.text.splitlines()
            if line.startswith("data: {")
        )
        message = chunk["choices"][0]["delta"]
        assert message["tool_calls"][0]["index"] == 0
    else:
        message = reply.json()["choices"][0]["message"]
    for key, value in reasoning.items():
        assert message[key] == value

    # Frameworks may retain tool calls while omitting provider extension fields.
    assistant = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    body = _body(messages=[assistant])
    await bridge.complete(body, rollout_id=ROLLOUT_ID)
    assert fake.completion_kwargs[-1]["messages"][0] == {**assistant, **reasoning}
    assert body["messages"] == [assistant]
    assert "thinking_blocks" not in assistant

    # Neither another rollout nor a finished rollout can receive cached context.
    await bridge.complete(body, rollout_id="b" * 32)
    assert fake.completion_kwargs[-1]["messages"][0] == assistant
    bridge.collect_tokens(ROLLOUT_ID)
    await bridge.complete(body, rollout_id=ROLLOUT_ID)
    assert fake.completion_kwargs[-1]["messages"][0] == assistant
    bridge.discard(ROLLOUT_ID)
    assert ROLLOUT_ID not in bridge._reasoning


async def test_reasoning_replay_preserves_explicit_client_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    bridge._reasoning[ROLLOUT_ID] = {
        "call_1": {
            "reasoning_content": "stored",
            "thinking_blocks": [{"type": "thinking", "signature": "stored"}],
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "stored"}],
        }
    }
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1"}],
        "reasoning_content": "supplied",
        "thinking_blocks": [{"type": "thinking", "signature": "supplied"}],
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "supplied"}],
    }
    await bridge.complete(_body(messages=[assistant]), rollout_id=ROLLOUT_ID)
    assert fake.completion_kwargs[-1]["messages"][0] == assistant


@pytest.mark.parametrize("mode", ["json", "stream", "keepalive"])
async def test_provider_errors_preserve_4xx_and_redact_all_response_modes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    provider_key = "local-provider-test-secret"
    ambient_key = "ambient-provider-test-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", ambient_key)
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.01)
    upstream = httpx.Response(
        422,
        request=httpx.Request("POST", "https://provider.example/v1/chat/completions"),
        json={
            "error": {
                "message": (
                    f"custom_option is invalid: {provider_key}, {ambient_key}, "
                    f"{BRIDGE_TOKEN}"
                )
            }
        },
    )
    failure = InternalServerError("LiteLLM rewrote the provider status")
    failure.__cause__ = httpx.HTTPStatusError(
        "unprocessable", request=upstream.request, response=upstream
    )
    _install(
        monkeypatch,
        _SlowLiteLLM(
            delay=0.05 if mode == "keepalive" else 0,
            completion_error=failure,
        ),
    )
    bridge = LiteLLMBridge(model="anthropic/claude-test", api_key=provider_key)
    async with _client(bridge, non_stream_keepalive=mode == "keepalive") as client:
        response = await client.post(
            _url(), json=_body(stream=mode == "stream"), headers=_auth()
        )
    assert response.status_code == (422 if mode == "json" else 200)
    if mode == "stream":
        payload = next(
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        )
    else:
        payload = response.json()
    assert payload["error"]["code"] == 422
    assert "custom_option is invalid" in payload["error"]["message"]
    for secret in (provider_key, ambient_key, BRIDGE_TOKEN):
        assert secret not in response.text
        assert secret not in caplog.text


async def test_invalid_bridge_controls_are_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    async with _client(LiteLLMBridge(model="anthropic/claude-test")) as client:
        response = await client.post(
            _url(), json=_body(extra_body={"api_key": "untrusted"}), headers=_auth()
        )
    assert response.status_code == 400
    assert "api_key" in response.json()["error"]["message"]
    assert fake.completion_kwargs == []


@pytest.mark.parametrize(
    ("field", "empty", "cached"),
    [
        ("thinking_blocks", None, [{"type": "thinking", "signature": "signed"}]),
        ("thinking_blocks", [], [{"type": "thinking", "signature": "signed"}]),
        (
            "reasoning_details",
            None,
            [{"type": "reasoning.encrypted", "data": "opaque"}],
        ),
        ("reasoning_details", [], [{"type": "reasoning.encrypted", "data": "opaque"}]),
        ("reasoning_content", None, "Use the tool."),
        ("reasoning_content", "", "Use the tool."),
    ],
)
async def test_reasoning_replay_restores_empty_client_fields(
    monkeypatch: pytest.MonkeyPatch, field: str, empty: Any, cached: Any
) -> None:
    fake = _FakeLiteLLM()
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test")
    bridge._reasoning[ROLLOUT_ID] = {"call_1": {field: cached}}
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1"}],
        field: empty,
    }
    await bridge.complete(_body(messages=[assistant]), rollout_id=ROLLOUT_ID)
    assert fake.completion_kwargs[-1]["messages"][0][field] == cached
    assert assistant[field] == empty


@pytest.mark.parametrize("mode", ["json", "stream", "keepalive"])
@pytest.mark.parametrize(
    "api_base",
    [
        "https://endpoint-secret@provider.example/v1",
        "https://provider.example/v1?token=endpoint-secret",
    ],
)
async def test_provider_error_redacts_credential_bearing_base_url(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    api_base: str,
) -> None:
    monkeypatch.setattr(llm_bridge, "_NON_STREAM_GRACE_SEC", 0.01)
    _install(
        monkeypatch,
        _SlowLiteLLM(
            delay=0.05 if mode == "keepalive" else 0,
            completion_error=InternalServerError(f"Connection failed: {api_base}"),
        ),
    )
    bridge = LiteLLMBridge(model="anthropic/claude-test", api_base=api_base)
    async with _client(bridge, non_stream_keepalive=mode == "keepalive") as client:
        response = await client.post(
            _url(), json=_body(stream=mode == "stream"), headers=_auth()
        )
    assert response.status_code == (502 if mode == "json" else 200)
    assert "Connection failed" in response.text
    assert "endpoint-secret" not in response.text
    assert "endpoint-secret" not in caplog.text


@pytest.mark.parametrize("during_provider_lookup", [False, True])
async def test_preflight_redacts_credential_bearing_base_url(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    during_provider_lookup: bool,
) -> None:
    api_base = "https://endpoint-secret@provider.example/v1"
    error = InternalServerError(f"Connection failed: {api_base}")
    fake = _FakeLiteLLM(
        provider_error=error if during_provider_lookup else None,
        completion_error=None if during_provider_lookup else error,
    )
    _install(monkeypatch, fake)
    bridge = LiteLLMBridge(model="anthropic/claude-test", api_base=api_base)
    if during_provider_lookup:
        with pytest.raises(RuntimeError, match="Invalid LiteLLM model format") as exc:
            await bridge.preflight_check()
        assert "endpoint-secret" not in str(exc.value)
    else:
        await bridge.preflight_check()
        assert "Connection failed" in caplog.text
    assert "endpoint-secret" not in caplog.text
