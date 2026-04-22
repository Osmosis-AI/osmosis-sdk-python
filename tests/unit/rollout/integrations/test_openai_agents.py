"""Tests for osmosis_ai.rollout.integrations.agents.openai_agents."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agents import function_tool
from agents.extensions.models.litellm_model import LitellmModel

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


class _FakeStreamingRunResult:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        events: list[dict[str, Any]] | None = None,
        exception: BaseException | None = None,
        final_output: Any = None,
    ) -> None:
        self._messages = messages
        self._events = events or []
        self.run_loop_exception = exception
        self.final_output = final_output

    async def stream_events(self):
        for event in self._events:
            yield event

    def to_input_list(self):
        return self._messages

    def final_output_as(self, _: type[Any]) -> Any:
        return self.final_output


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _ChunkSSEServer:
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
                data = _sse_body(owner._payloads[idx])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
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


def _chunk_payload(
    *,
    content: str | None = None,
    finish_reason: str | None = "stop",
    role: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "osmosis-rollout",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage,
    }


@function_tool
def add(a: int, b: int) -> str:
    return str(a + b)


class TestOsmosisOpenAIAgent:
    def test_requires_active_rollout_context(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(RuntimeError, match="active RolloutContext"):
            OsmosisOpenAIAgent(name="x")

    def test_builds_litellm_model_from_context(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        agent = OsmosisOpenAIAgent(name="main")

        assert isinstance(agent.model, LitellmModel)
        assert agent.model.model == "openai/osmosis-rollout"
        assert agent.model.base_url == "http://controller:9"
        assert agent.model.api_key == "test-key"

    def test_rejects_instructions_argument(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(TypeError, match="does not accept instructions"):
            OsmosisOpenAIAgent(name="main", instructions="do stuff")

    def test_rejects_prompt_argument(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(TypeError, match="does not accept prompt"):
            OsmosisOpenAIAgent(name="main", prompt={"id": "rollout-prompt"})

    def test_rejects_positional_instructions_argument(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(TypeError, match="does not accept instructions"):
            OsmosisOpenAIAgent("main", None, [], [], {}, "do stuff")

    def test_rejects_positional_prompt_argument(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(TypeError, match="does not accept prompt"):
            OsmosisOpenAIAgent("main", None, [], [], {}, None, {"id": "rollout-prompt"})

    def test_propagates_none_api_key_when_missing(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        ctx = RolloutContext(
            chat_completions_url="http://controller:9",
            api_key=None,
            rollout_id="r1",
        )

        with ctx:
            agent = OsmosisOpenAIAgent(name="main")

        assert isinstance(agent.model, LitellmModel)
        assert agent.model.api_key is None

    def test_user_supplied_model_is_respected(self, rollout_context):
        from agents.models.interface import Model

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        custom_model = MagicMock(spec=Model, name="UserModel")
        agent = OsmosisOpenAIAgent(name="main", model=custom_model)

        assert agent.model is custom_model

    def test_user_supplied_model_settings_are_preserved_verbatim(self, rollout_context):
        """Mirror Strands: no automatic extra_body injection — what the
        user passes is what hits the wire."""
        from agents.model_settings import ModelSettings

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        agent = OsmosisOpenAIAgent(
            name="main",
            model_settings=ModelSettings(extra_body={"existing": "value"}),
        )

        assert agent.model_settings.extra_body == {"existing": "value"}


class TestOsmosisOpenAIAgentRun:
    async def test_injects_headers_via_context_var(self, rollout_context):
        from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        seen_headers = {}

        def fake_run_streamed(agent, input, *, run_config, **kwargs):
            seen_headers.update(HEADERS_OVERRIDE.get() or {})
            return _FakeStreamingRunResult(
                [{"role": "assistant", "content": "done"}],
                events=[{"type": "response.created"}],
                final_output="done",
            )

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            side_effect=fake_run_streamed,
        ):
            agent = OsmosisOpenAIAgent(name="main")
            result = await agent.run("hi")

        assert result.final_output == "done"
        assert seen_headers["x-rollout-id"] == "rollout-xyz"
        assert seen_headers["x-sample-id"] == "main"

    async def test_records_sample_after_run(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        messages = [{"role": "assistant", "content": "42"}]

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            return_value=_FakeStreamingRunResult(messages),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi")

        samples = rollout_context.get_samples()
        assert "main" in samples
        assert samples["main"].messages == messages

    async def test_resolves_collision_with_suffix_and_warns(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "ok"}]
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("turn 1")
            with pytest.warns(RuntimeWarning, match="already used"):
                await agent.run("turn 2")

        samples = rollout_context.get_samples()
        assert "main" in samples
        other_ids = [sid for sid in samples if sid != "main"]
        assert len(other_ids) == 1
        assert other_ids[0].startswith("main-")

    async def test_explicit_sample_id_is_respected(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "ok"}]
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi", sample_id="explicit-id")

        samples = rollout_context.get_samples()
        assert "explicit-id" in samples
        assert "main" not in samples

    async def test_raises_outside_rollout_context(self, rollout_context):
        from osmosis_ai.rollout.context import rollout_contextvar
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        agent = OsmosisOpenAIAgent(name="main")

        token = rollout_contextvar.set(None)
        try:
            with pytest.raises(RuntimeError, match="requires an active RolloutContext"):
                await agent.run("hi")
        finally:
            rollout_contextvar.reset(token)

    async def test_forces_tracing_disabled(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        captured = {}

        def fake_run_streamed(agent, input, *, run_config, **kwargs):
            captured["run_config"] = run_config
            return _FakeStreamingRunResult([{"role": "assistant", "content": "x"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            side_effect=fake_run_streamed,
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi")

        assert captured["run_config"].tracing_disabled is True

    async def test_forwards_runner_kwargs(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        captured = {}

        def fake_run_streamed(agent, input, *, run_config, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeStreamingRunResult([{"role": "assistant", "content": "x"}])

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            side_effect=fake_run_streamed,
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi", max_turns=8)

        assert captured["kwargs"]["max_turns"] == 8

    async def test_raises_background_stream_exception(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run_streamed",
            return_value=_FakeStreamingRunResult(
                [{"role": "assistant", "content": "x"}],
                exception=RuntimeError("stream failed"),
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            with pytest.raises(RuntimeError, match="stream failed"):
                await agent.run("hi")

        assert rollout_context.get_samples() == {}


class TestOsmosisOpenAIAgentStreamingIntegration:
    async def test_run_streamed_consumes_chunk_sse_response(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        seen_requests: list[dict[str, Any]] = []
        server = _ChunkSSEServer(
            [
                _chunk_payload(
                    content="hi",
                    finish_reason="stop",
                    role="assistant",
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
                agent = OsmosisOpenAIAgent(name="main")
                result = await agent.run("hello")

            samples = ctx.get_samples()
            body = seen_requests[0]["body"]
            assert body["stream"] is True
            # Match the Strands wire contract: no extra_body / multi_turn_mode.
            assert "multi_turn_mode" not in body
            assert seen_requests[0]["headers"]["x-rollout-id"] == "rollout-xyz"
            assert seen_requests[0]["headers"]["x-sample-id"] == "main"
            assert result.final_output_as(str) == "hi"
            assert samples["main"].messages == result.to_input_list()
        finally:
            server.close()

    async def test_run_streamed_handles_tool_call_loop_over_sse(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        seen_requests: list[dict[str, Any]] = []
        server = _ChunkSSEServer(
            [
                _chunk_payload(
                    finish_reason="tool_calls",
                    role="assistant",
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
                _chunk_payload(
                    content="The answer is 3",
                    finish_reason="stop",
                    role="assistant",
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
                agent = OsmosisOpenAIAgent(name="main", tools=[add])
                result = await agent.run("hello")

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
