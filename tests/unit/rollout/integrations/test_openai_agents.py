"""Tests for osmosis_ai.rollout.integrations.agents.openai_agents."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import Agent, ModelSettings, RunConfig, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model, ModelProvider

from osmosis_ai.rollout.context import RolloutContext

pytestmark = pytest.mark.filterwarnings(
    "ignore:Pydantic serializer warnings:UserWarning"
)


@pytest.fixture
def rollout_context():
    ctx = RolloutContext(
        chat_completions_url="http://controller:9",
        api_key="test-key",
        rollout_id="rollout-xyz",
    )
    with ctx:
        yield ctx


class _FakeRunResult:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        final_output: Any = None,
    ) -> None:
        self._messages = messages
        self.final_output = final_output

    def to_input_list(self):
        return self._messages

    def final_output_as(self, _: type[Any]) -> Any:
        return self.final_output


class _FakeStreamingRunResult(_FakeRunResult):
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        events: list[dict[str, Any]] | None = None,
        exception: BaseException | None = None,
        run_loop_exception: BaseException | None = None,
        final_output: Any = None,
    ) -> None:
        super().__init__(messages, final_output=final_output)
        self._events = events or []
        self._stream_exception = exception
        self.run_loop_exception = run_loop_exception

    async def stream_events(self):
        for event in self._events:
            yield event
        if self._stream_exception is not None:
            raise self._stream_exception


class _FakeModelProvider(ModelProvider):
    def __init__(self, model: Model) -> None:
        self.model = model
        self.model_names: list[str | None] = []

    def get_model(self, model_name: str | None) -> Model:
        self.model_names.append(model_name)
        return self.model

    async def aclose(self) -> None:
        return None


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _ChatCompletionsServer:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        seen_requests: list[dict[str, Any]],
    ) -> None:
        self._payloads = payloads
        self._seen_requests = seen_requests
        self._request_count = 0
        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _make_handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8") or "{}")
                owner._seen_requests.append(
                    {
                        "path": self.path,
                        "body": body,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                idx = min(owner._request_count, len(owner._payloads) - 1)
                owner._request_count += 1
                payload = owner._payloads[idx]
                if body.get("stream"):
                    data = _sse_body(payload)
                    content_type = "text/event-stream"
                else:
                    data = json.dumps(payload).encode()
                    content_type = "application/json"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def log_message(self, *args: Any) -> None:
                return

        return Handler


def _sse_body(payload: dict[str, Any]) -> bytes:
    return (f": ping\n\ndata: {json.dumps(payload)}\n\ndata: [DONE]\n\n").encode()


def _completion_payload(
    *,
    content: str | None = None,
    finish_reason: str | None = "stop",
    role: str | None = "assistant",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": role, "content": content or ""}
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        delta["tool_calls"] = tool_calls

    choice: dict[str, Any] = {
        "index": 0,
        "finish_reason": finish_reason,
        "logprobs": None,
    }
    if stream:
        choice["delta"] = delta
    else:
        choice["message"] = message

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": "osmosis-rollout",
        "choices": [choice],
        "usage": usage,
    }


@function_tool
def add(a: int, b: int) -> str:
    return str(a + b)


class TestOpenAIAgentsRunner:
    async def test_delegates_without_rollout_context(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        agent = Agent(name="main")
        expected = _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run",
            new=AsyncMock(return_value=expected),
        ) as run:
            result = await Runner.run(agent, "hi")

        assert result is expected
        run.assert_awaited_once()

    def test_run_streamed_delegates_without_rollout_context(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        expected = _FakeStreamingRunResult([{"role": "assistant", "content": "ok"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=expected,
        ) as run_streamed:
            result = Runner.run_streamed(Agent(name="main"), "hi")

        assert result is expected
        run_streamed.assert_called_once()

    async def test_run_streamed_records_after_stream_completes(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        messages = [{"role": "assistant", "content": "ok"}]

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(messages, events=[{"type": "done"}]),
        ):
            result = Runner.run_streamed(Agent(name="main"), "hi")
            assert rollout_context.get_samples() == {}
            async for _ in result.stream_events():
                pass

        assert rollout_context.get_samples()["main"].messages == messages

    async def test_run_streamed_respects_agent_model_in_rollout_context(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        custom_model = MagicMock(spec=Model, name="UserModel")
        model_provider = _FakeModelProvider(custom_model)
        captured = {}

        def fake_run(agent, input, *, run_config, **kwargs):
            captured["agent_model"] = agent.model
            captured["resolved_model"] = run_config.model_provider.get_model(
                agent.model
            )
            return _FakeStreamingRunResult(
                [{"role": "assistant", "content": "ok"}],
                events=[{"type": "done"}],
            )

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            side_effect=fake_run,
        ):
            result = Runner.run_streamed(
                Agent(name="main", model="gpt-4o"),
                "hi",
                run_config=RunConfig(model_provider=model_provider),
            )
            async for _ in result.stream_events():
                pass

        assert captured["agent_model"] == "gpt-4o"
        assert captured["resolved_model"] is custom_model
        assert model_provider.model_names == ["gpt-4o"]
        assert rollout_context.get_samples()["main"].messages == [
            {"role": "assistant", "content": "ok"}
        ]

    async def test_agent_can_be_constructed_outside_rollout_context(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        agent = Agent(name="main", instructions="do stuff")
        messages = [{"role": "assistant", "content": "ok"}]

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(messages, final_output="ok"),
        ):
            result = await Runner.run(agent, "hi")

        assert result.final_output == "ok"
        assert rollout_context.get_samples()["main"].messages == messages

    async def test_builds_litellm_model_from_context(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        captured = {}

        def fake_run(agent, input, *, run_config, **kwargs):
            captured["run_config"] = run_config
            captured["model"] = run_config.model_provider.get_model(None)
            return _FakeStreamingRunResult([{"role": "assistant", "content": "ok"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            side_effect=fake_run,
        ):
            await Runner.run(Agent(name="main"), "hi")

        model = captured["model"]
        assert isinstance(model, LitellmModel)
        assert model.model == "openai/osmosis-rollout"
        assert model.base_url == "http://controller:9"
        assert model.api_key == "test-key"

    async def test_agent_model_is_respected_in_rollout_context(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        captured = {}

        async def fake_run(agent, input, *, run_config, **kwargs):
            captured["agent"] = agent
            captured["run_config"] = run_config
            return _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with (
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run",
                side_effect=fake_run,
            ) as run,
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed"
            ) as run_streamed,
        ):
            await Runner.run(Agent(name="main", model="gpt-4o"), "hi")

        run.assert_awaited_once()
        run_streamed.assert_not_called()
        assert captured["agent"].model == "gpt-4o"
        assert captured["run_config"].model is None
        assert rollout_context.get_samples()["main"].messages == [
            {"role": "assistant", "content": "ok"}
        ]

    async def test_none_model_provider_uses_default_fallback(
        self, rollout_context, monkeypatch
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}

        async def fake_run(agent, input, *, run_config, **kwargs):
            captured["resolved_model"] = run_config.model_provider.get_model(
                agent.model
            )
            return _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run",
            side_effect=fake_run,
        ):
            await Runner.run(
                Agent(name="main", model="gpt-4o"),
                "hi",
                run_config=RunConfig(model_provider=None),
            )

        assert captured["resolved_model"] is not None

    async def test_user_supplied_run_config_model_is_respected(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        custom_model = MagicMock(spec=Model, name="UserModel")
        captured = {}

        async def fake_run(agent, input, *, run_config, **kwargs):
            captured["run_config"] = run_config
            return _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with (
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run",
                side_effect=fake_run,
            ) as run,
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed"
            ) as run_streamed,
        ):
            await Runner.run(
                Agent(name="main"),
                "hi",
                run_config=RunConfig(model=custom_model),
            )

        run.assert_awaited_once()
        run_streamed.assert_not_called()
        assert captured["run_config"].model is custom_model

    async def test_injects_headers_on_rollout_model_only(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        seen_headers = {}

        def fake_run(agent, input, *, run_config, **kwargs):
            model = run_config.model_provider.get_model(None)
            seen_headers.update(model._merge_headers(ModelSettings()))
            return _FakeStreamingRunResult([{"role": "assistant", "content": "done"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            side_effect=fake_run,
        ):
            await Runner.run(Agent(name="main"), "hi")

        assert seen_headers["x-rollout-id"] == "rollout-xyz"
        assert seen_headers["x-sample-id"] == "main"

    async def test_resolves_collision_with_suffix_and_warns(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "ok"}]
            ),
        ):
            agent = Agent(name="main")
            await Runner.run(agent, "turn 1")
            with pytest.warns(RuntimeWarning, match="already used"):
                await Runner.run(agent, "turn 2")

        samples = rollout_context.get_samples()
        assert "main" in samples
        other_ids = [sid for sid in samples if sid != "main"]
        assert len(other_ids) == 1
        assert other_ids[0].startswith("main-")

    async def test_explicit_sample_id_is_respected(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "ok"}]
            ),
        ):
            await Runner.run(Agent(name="main"), "hi", sample_id="explicit-id")

        samples = rollout_context.get_samples()
        assert "explicit-id" in samples
        assert "main" not in samples

    async def test_forces_tracing_disabled(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        captured = {}

        def fake_run(agent, input, *, run_config, **kwargs):
            captured["run_config"] = run_config
            return _FakeStreamingRunResult([{"role": "assistant", "content": "x"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            side_effect=fake_run,
        ):
            await Runner.run(Agent(name="main"), "hi")

        assert captured["run_config"].tracing_disabled is True

    async def test_forwards_runner_kwargs(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        captured = {}

        def fake_run(agent, input, *, run_config, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeStreamingRunResult([{"role": "assistant", "content": "x"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            side_effect=fake_run,
        ):
            await Runner.run(Agent(name="main"), "hi", max_turns=8)

        assert captured["kwargs"]["max_turns"] == 8

    async def test_does_not_record_sample_when_stream_raises(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "x"}],
                exception=RuntimeError("stream failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="stream failed"):
                await Runner.run(Agent(name="main"), "hi")

        assert rollout_context.get_samples() == {}

    async def test_does_not_record_sample_when_run_loop_exception_is_stored(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.OpenAIRunner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "x"}],
                events=[{"type": "done"}],
                run_loop_exception=RuntimeError("run loop failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="run loop failed"):
                await Runner.run(Agent(name="main"), "hi")

        assert rollout_context.get_samples() == {}

    async def test_run_consumes_chunk_sse_response(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        seen_requests: list[dict[str, Any]] = []
        server = _ChatCompletionsServer(
            [
                _completion_payload(
                    content="hi",
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                )
            ],
            seen_requests,
        )
        ctx = RolloutContext(
            chat_completions_url=server.url,
            api_key="test-key",
            rollout_id="rollout-xyz",
        )

        try:
            with ctx:
                result = await Runner.run(Agent(name="main"), "hello")

            samples = ctx.get_samples()
            body = seen_requests[0]["body"]
            assert body["stream"] is True
            assert "multi_turn_mode" not in body
            assert seen_requests[0]["headers"]["x-rollout-id"] == "rollout-xyz"
            assert seen_requests[0]["headers"]["x-sample-id"] == "main"
            assert result.final_output_as(str) == "hi"
            assert samples["main"].messages == result.to_input_list()
        finally:
            server.close()

    async def test_run_preserves_system_and_user_prompt_roles(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        seen_requests: list[dict[str, Any]] = []
        server = _ChatCompletionsServer(
            [
                _completion_payload(
                    content="hi",
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                )
            ],
            seen_requests,
        )
        ctx = RolloutContext(
            chat_completions_url=server.url,
            api_key="test-key",
            rollout_id="rollout-xyz",
        )
        prompt = [
            {"role": "system", "content": "Follow the training rubric."},
            {"role": "user", "content": "Say hi."},
        ]

        try:
            with ctx:
                await Runner.run(Agent(name="main"), prompt)

            messages = seen_requests[0]["body"]["messages"]
            assert messages[0] == {
                "role": "system",
                "content": "Follow the training rubric.",
            }
            assert messages[1] == {"role": "user", "content": "Say hi."}
            assert seen_requests[0]["body"]["stream"] is True
        finally:
            server.close()

    async def test_run_handles_tool_call_loop_over_sse(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import Runner

        seen_requests: list[dict[str, Any]] = []
        server = _ChatCompletionsServer(
            [
                _completion_payload(
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 1, "b": 2}',
                            },
                        }
                    ],
                    usage={
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                ),
                _completion_payload(
                    content="The answer is 3",
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 4,
                        "completion_tokens": 4,
                        "total_tokens": 8,
                    },
                ),
            ],
            seen_requests,
        )
        ctx = RolloutContext(
            chat_completions_url=server.url,
            api_key="test-key",
            rollout_id="rollout-xyz",
        )

        try:
            with ctx:
                result = await Runner.run(Agent(name="main", tools=[add]), "hello")

            messages = result.to_input_list()
            assert len(seen_requests) == 2
            assert seen_requests[0]["body"]["stream"] is True
            assert seen_requests[1]["body"]["messages"][-2]["role"] == "assistant"
            assert seen_requests[1]["body"]["messages"][-1]["role"] == "tool"
            assert seen_requests[1]["body"]["messages"][-1]["content"] == "3"
            assert result.final_output_as(str) == "The answer is 3"
            assert any(
                item.get("type") == "function_call" and item.get("name") == "add"
                for item in messages
                if isinstance(item, dict)
            )
            assert any(
                item.get("type") == "function_call_output" and item.get("output") == "3"
                for item in messages
                if isinstance(item, dict)
            )
            assert ctx.get_samples()["main"].messages == messages
        finally:
            server.close()
