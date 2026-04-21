"""Tests for LiteLLMProxy — lightweight LLM proxy with token counting."""

from __future__ import annotations

import pytest

from osmosis_ai.eval.llm_proxy import LiteLLMProxy


def test_proxy_url_before_start():
    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    with pytest.raises(RuntimeError, match="not been started"):
        _ = proxy.url


def test_proxy_collect_tokens_empty():
    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    assert proxy.collect_tokens("nonexistent") == 0


def test_proxy_collect_tokens_accumulates():
    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    proxy._tokens["r1"] = 100
    proxy._tokens["r1"] += 50
    assert proxy.collect_tokens("r1") == 150
    # Second call returns 0 (popped)
    assert proxy.collect_tokens("r1") == 0


def test_proxy_systemic_error_initially_empty():
    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    assert proxy.collect_systemic_error("any") is None


def test_proxy_systemic_exceptions_list():
    assert "AuthenticationError" in LiteLLMProxy.SYSTEMIC_EXCEPTIONS
    assert "BudgetExceededError" in LiteLLMProxy.SYSTEMIC_EXCEPTIONS
    assert "NotFoundError" in LiteLLMProxy.SYSTEMIC_EXCEPTIONS


def test_proxy_trace_dir_none_by_default():
    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    assert proxy.trace_dir is None


def test_proxy_trace_dir_set():
    from pathlib import Path

    proxy = LiteLLMProxy(model="openai/gpt-5-mini", trace_dir="/tmp/traces")
    assert proxy.trace_dir == Path("/tmp/traces")


async def test_preflight_check_bad_model_format(monkeypatch):
    """preflight_check raises CLIError on invalid model format."""
    from osmosis_ai.cli.errors import CLIError

    def fake_get_llm_provider(model, api_base=None):
        raise Exception("Bad model format")

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "get_llm_provider": staticmethod(fake_get_llm_provider),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="gpt5-bad")
    with pytest.raises(CLIError, match="Invalid LiteLLM model format"):
        await proxy.preflight_check()


async def test_preflight_check_auth_failure(monkeypatch):
    """preflight_check raises CLIError on authentication error."""
    from osmosis_ai.cli.errors import CLIError

    class FakeAuthError(Exception):
        pass

    FakeAuthError.__name__ = "AuthenticationError"

    async def fake_acompletion(**kwargs):
        raise FakeAuthError("Invalid API key")

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "get_llm_provider": staticmethod(
                lambda model, api_base=None: ("openai", "gpt-5-mini", None, None)
            ),
            "acompletion": staticmethod(fake_acompletion),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    with pytest.raises(CLIError, match="LLM preflight check failed"):
        await proxy.preflight_check()


async def test_handle_request_systemic_error_sets_flag(monkeypatch):
    """_handle_request sets systemic_error flag on AuthenticationError."""

    class FakeAuthError(Exception):
        pass

    FakeAuthError.__name__ = "AuthenticationError"

    async def fake_acompletion(**kwargs):
        raise FakeAuthError("Invalid API key")

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "acompletion": staticmethod(fake_acompletion),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    assert proxy.collect_systemic_error("r1") is None

    with pytest.raises(FakeAuthError):
        await proxy._handle_request(
            "r1", {"messages": [{"role": "user", "content": "hi"}]}
        )

    err = proxy.collect_systemic_error("r1")
    assert err is not None
    assert "Invalid API key" in err
    # Second collect returns None (popped)
    assert proxy.collect_systemic_error("r1") is None


async def test_handle_request_writes_trace(monkeypatch, tmp_path):
    """_handle_request writes JSONL trace when trace_dir is set."""
    from unittest.mock import MagicMock

    usage = MagicMock()
    usage.total_tokens = 42
    usage.prompt_tokens = 20
    usage.completion_tokens = 22

    message = MagicMock()
    message.model_dump.return_value = {"role": "assistant", "content": "ok"}

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model_dump.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 22, "total_tokens": 42},
    }

    async def fake_acompletion(**kwargs):
        return response

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "acompletion": staticmethod(fake_acompletion),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="openai/gpt-5-mini", trace_dir=str(tmp_path))
    await proxy._handle_request("r1", {"messages": [{"role": "user", "content": "hi"}]})

    trace_file = tmp_path / "r1.jsonl"
    assert trace_file.exists()
    content = trace_file.read_text()
    assert "llm_call" in content


async def test_stream_response_mid_stream_error(monkeypatch):
    """Mid-stream exception yields error frame + [DONE] instead of broken SSE."""
    import json
    from unittest.mock import MagicMock

    async def _exploding_iterator():
        chunk = MagicMock()
        chunk.usage = None
        chunk.model_dump.return_value = {"choices": [{"delta": {"content": "hi"}}]}
        yield chunk
        raise RuntimeError("connection reset by peer")

    async def fake_acompletion(**kwargs):
        return _exploding_iterator()

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "acompletion": staticmethod(fake_acompletion),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    frames = []
    async for frame in proxy._stream_response(
        "r1", {"messages": [{"role": "user", "content": "hi"}]}
    ):
        frames.append(frame)

    # Should have: one data chunk, one error frame, one [DONE]
    assert len(frames) == 3
    assert '"delta"' in frames[0]
    error_payload = json.loads(frames[1].removeprefix("data: ").strip())
    assert error_payload["error"]["type"] == "RuntimeError"
    assert "connection reset" in error_payload["error"]["message"]
    assert frames[-1] == "data: [DONE]\n\n"


async def test_stream_response_mid_stream_systemic_error(monkeypatch):
    """Mid-stream systemic error is recorded in _systemic_errors."""

    class FakeAuthError(Exception):
        pass

    FakeAuthError.__name__ = "AuthenticationError"

    async def _auth_fail_iterator():
        raise FakeAuthError("token revoked mid-stream")
        yield  # make it an async generator

    async def fake_acompletion(**kwargs):
        return _auth_fail_iterator()

    import osmosis_ai.eval.llm_proxy as _mod

    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "suppress_debug_info": False,
            "acompletion": staticmethod(fake_acompletion),
        },
    )()
    monkeypatch.setattr(_mod, "_get_litellm", lambda: fake_litellm)

    proxy = LiteLLMProxy(model="openai/gpt-5-mini")
    frames = []
    async for frame in proxy._stream_response(
        "r1", {"messages": [{"role": "user", "content": "hi"}]}
    ):
        frames.append(frame)

    # Error frame + [DONE]
    assert frames[-1] == "data: [DONE]\n\n"
    err = proxy.collect_systemic_error("r1")
    assert err is not None
    assert "token revoked" in err
