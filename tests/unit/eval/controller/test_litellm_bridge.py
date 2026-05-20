import subprocess
import sys
from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall

from osmosis_ai.eval.controller.litellm_bridge import (
    LiteLLMBridge,
    build_sse_error_event,
    compact_json,
    model_response_to_payload,
)


def test_importing_bridge_does_not_eagerly_import_litellm() -> None:
    script = "\n".join(
        [
            "import sys",
            "import osmosis_ai.eval.controller.litellm_bridge",
            "assert 'litellm' not in sys.modules, 'litellm was eagerly imported'",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_build_kwargs_uses_configured_model_and_filters_ignored_fields() -> None:
    bridge = LiteLLMBridge(
        model="openai/real-model", api_key="secret", base_url="https://provider"
    )
    kwargs = bridge.build_kwargs(
        {
            "model": "openai/osmosis-rollout",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 5,
            "stop": ["ignored"],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "stream_options": {"include_usage": True},
        }
    )

    assert kwargs == {
        "model": "openai/real-model",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "secret",
        "api_base": "https://provider",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 5,
        "drop_params": True,
    }


def test_build_kwargs_forwards_tools() -> None:
    bridge = LiteLLMBridge(model="openai/real-model")

    assert bridge.build_kwargs(
        {
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ) == {
        "model": "openai/real-model",
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "drop_params": True,
    }


def test_build_kwargs_lets_litellm_drop_provider_unsupported_params() -> None:
    bridge = LiteLLMBridge(model="openai/gpt-5-mini")

    kwargs = bridge.build_kwargs(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 1.0,
        }
    )

    assert kwargs["top_p"] == 1.0
    assert kwargs["drop_params"] is True


def test_model_response_payload_mirrors_request_model_and_stream_delta_shape() -> None:
    response = SimpleNamespace(
        id="chatcmpl-1",
        created=123,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(content="hello", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )

    payload = model_response_to_payload(
        response, request_model="openai/osmosis-rollout", stream=True
    )

    assert payload["model"] == "openai/osmosis-rollout"
    assert payload["stream"] is True
    assert payload["choices"][0]["delta"] == {"content": "hello"}
    assert "message" not in payload["choices"][0]
    assert payload["usage"]["total_tokens"] == 5


def test_model_response_payload_non_stream_message_shape() -> None:
    response = {
        "id": "chatcmpl-1",
        "created": 123,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "hello", "tool_calls": None},
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }

    payload = model_response_to_payload(
        response, request_model="openai/osmosis-rollout", stream=False
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "openai/osmosis-rollout"
    assert "stream" not in payload
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
    }
    assert payload["usage"]["total_tokens"] == 5


def test_model_response_payload_serializes_real_litellm_tool_calls() -> None:
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "lookup", "arguments": "{}"},
    )
    response = SimpleNamespace(
        id="chatcmpl-1",
        created=123,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
            )
        ],
        usage=None,
    )

    payload = model_response_to_payload(
        response, request_model="openai/osmosis-rollout", stream=False
    )

    assert payload["choices"][0]["message"]["tool_calls"] == [
        {
            "function": {"arguments": "{}", "name": "lookup"},
            "id": "call_1",
            "type": "function",
        }
    ]
    compact_json(payload)


def test_model_response_payload_serializes_stream_tool_call_delta() -> None:
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "lookup", "arguments": "{}"},
    )
    response = SimpleNamespace(
        id="chatcmpl-1",
        created=123,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=None,
                delta=SimpleNamespace(content="", tool_calls=[tool_call]),
            )
        ],
        usage=None,
    )

    payload = model_response_to_payload(
        response, request_model="openai/osmosis-rollout", stream=True
    )

    assert payload["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "function": {"arguments": "{}", "name": "lookup"},
            "id": "call_1",
            "type": "function",
        }
    ]
    compact_json(payload)


def test_stream_tool_call_delta_adds_missing_indexes_for_multiple_calls() -> None:
    tool_calls = [
        ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={"name": "lookup", "arguments": "{}"},
        ),
        ChatCompletionMessageToolCall(
            id="call_2",
            type="function",
            function={"name": "search", "arguments": '{"q":"x"}'},
        ),
    ]
    response = SimpleNamespace(
        id="chatcmpl-1",
        created=123,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=None,
                delta=SimpleNamespace(content="", tool_calls=tool_calls),
            )
        ],
        usage=None,
    )

    payload = model_response_to_payload(
        response, request_model="openai/osmosis-rollout", stream=True
    )

    assert [item["index"] for item in payload["choices"][0]["delta"]["tool_calls"]] == [
        0,
        1,
    ]
    compact_json(payload)


def test_sse_error_event_is_compact_json() -> None:
    event = build_sse_error_event("bad key")

    assert event == 'event: error\ndata: {"error":"bad key"}\n\n'


@pytest.mark.asyncio
async def test_completion_records_systemic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAuthError(Exception):
        pass

    FakeAuthError.__name__ = "AuthenticationError"

    async def fake_acompletion(**kwargs):
        raise FakeAuthError("invalid key")

    fake_litellm = SimpleNamespace(
        acompletion=fake_acompletion, suppress_debug_info=False
    )
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.litellm_bridge._get_litellm",
        lambda: fake_litellm,
    )

    bridge = LiteLLMBridge(model="openai/real-model")

    with pytest.raises(FakeAuthError):
        await bridge.complete(
            "r1", "s1", {"messages": [{"role": "user", "content": "hi"}]}
        )

    assert bridge.collect_systemic_error("r1") == "invalid key"


@pytest.mark.asyncio
async def test_completion_records_tokens_and_returns_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="chatcmpl-1",
        created=123,
        choices=[],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )

    async def fake_acompletion(**kwargs):
        return response

    fake_litellm = SimpleNamespace(acompletion=fake_acompletion)
    monkeypatch.setattr(
        "osmosis_ai.eval.controller.litellm_bridge._get_litellm",
        lambda: fake_litellm,
    )

    bridge = LiteLLMBridge(model="openai/real-model")

    assert await bridge.complete("r1", "s1", {"messages": []}) is response
    assert bridge.collect_tokens("r1") == 5
    assert bridge.collect_tokens("r1") == 0
