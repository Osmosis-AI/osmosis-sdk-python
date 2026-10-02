from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agents import RunConfig, Runner
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from osmosis_ai.rollout.context import RolloutContext


@pytest.fixture
def rollout_context():
    ctx = RolloutContext(
        chat_completions_url="http://controller:9",
        api_key="test-key",
        rollout_id="rollout-xyz",
    )
    with ctx:
        yield ctx


class TestOpenAIAgentsIntegration:
    @pytest.mark.parametrize("limit", [-1, 0, 2, 10])
    async def test_preview_is_bounded_and_isolated_from_grading_history(
        self, rollout_context, limit
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": text}]}
            for text in ("first", "second", "third")
        ]
        items[0] = {"role": "assistant", "content": "first"}
        await session.add_items(items)

        preview = await rollout_context.sample_source.get_preview(limit)

        if limit <= 0:
            assert preview is None
        else:
            assert preview is not None
            assert preview.messages == items[-limit:]
            assert preview.trajectory_messages is not None
            assert preview.extra_fields == {
                "_preview_truncated": limit < 3,
                "_preview_turn": 1,
            }
            items[-1]["content"][0]["text"] = "live update"
            assert preview.messages[-1]["content"][0]["text"] == "third"
            assert preview.trajectory_messages[-1]["content"][0]["text"] == "third"
            preview.messages[-1]["content"][0]["text"] = "preview edit"

        sample = await rollout_context.get_sample()
        assert sample is not None
        assert len(sample.messages) == 3
        assert sample.messages == items
        assert sample.extra_fields == {}
        assert sample.messages[-1]["content"][0]["text"] == (
            "live update" if limit > 0 else "third"
        )

    async def test_responses_tool_history_does_not_claim_a_tail_local_turn(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        for index in range(110):
            await session.add_items(
                [
                    {
                        "type": "function_call",
                        "call_id": str(index),
                        "name": "shell",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": str(index),
                        "output": "ok",
                    },
                ]
            )
        sample = await rollout_context.sample_source.get_preview(100)
        assert sample.trajectory_messages
        assert sample.extra_fields["_preview_truncated"] is True
        assert sample.extra_fields["_preview_turn"] is None

    async def test_memory_session_registers_sample_source(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        items = [{"role": "user", "content": "hello"}]

        await session.add_items(items)

        sample = await rollout_context.get_sample()
        assert sample is not None
        assert sample.messages == items
        assert sample.trajectory_messages == items

    @pytest.mark.parametrize("preview", [False, True])
    async def test_trajectory_conversion_failure_keeps_native_messages(
        self, rollout_context, caplog, preview
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        items = [{"role": "user", "content": "hello"}]
        await session.add_items(items)
        sensitive_text = "sensitive-preview-content"

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.Converter.items_to_messages",
            side_effect=RuntimeError(sensitive_text),
        ):
            sample = (
                await rollout_context.sample_source.get_preview(10)
                if preview
                else await rollout_context.get_sample()
            )

        assert sample is not None
        assert sample.messages == items
        assert sample.trajectory_messages is None
        if preview:
            assert sensitive_text not in caplog.text
            assert not caplog.records
        else:
            assert any(
                "Failed to convert OpenAI Agents" in r.message
                and r.exc_info is not None
                for r in caplog.records
            )

    async def test_memory_session_raises_when_used_in_rollout_context_after_creation(
        self,
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        ctx = RolloutContext(
            chat_completions_url="http://controller:9",
            api_key="test-key",
            rollout_id="rollout-xyz",
        )

        with ctx:
            with pytest.raises(RuntimeError, match="not registered"):
                await session.add_items([{"role": "user", "content": "hello"}])

    def test_agent_swaps_placeholder_model_inside_rollout_context(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisAgent,
            OsmosisLitellmModel,
            OsmosisRolloutModel,
        )

        agent = OsmosisAgent(name="main", model=OsmosisRolloutModel())

        assert isinstance(agent.model, OsmosisLitellmModel)
        assert agent.model.model == "litellm_proxy/osmosis-rollout"
        assert agent.model.base_url == "http://controller:9"
        assert agent.model.api_key == "test-key"

    async def test_rollout_model_merges_headers_with_registered_session(
        self, rollout_context
    ):
        # The URL carries rollout identity, so no per-call routing headers
        # are stamped; the call just requires a registered sample source.
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisLitellmModel,
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        await session.get_items()
        model = OsmosisLitellmModel()

        headers = model._merge_headers(ModelSettings())

        assert "x-rollout-id" not in headers
        assert "x-sample-id" not in headers

    def test_rollout_model_requires_session_sample_id(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisLitellmModel,
        )

        model = OsmosisLitellmModel()

        with pytest.raises(RuntimeError, match="OsmosisMemorySession"):
            model._merge_headers(ModelSettings())

    async def test_get_response_aggregates_streaming_response(
        self, rollout_context, monkeypatch
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisLitellmModel,
        )

        model = OsmosisLitellmModel()
        output = [
            ResponseOutputMessage(
                id="msg_1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="hello",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        async def fake_stream_response(*_args, **_kwargs):
            yield SimpleNamespace(type="response.output_text.delta")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=output,
                    usage=SimpleNamespace(
                        input_tokens=3,
                        output_tokens=5,
                        total_tokens=8,
                        input_tokens_details=None,
                        output_tokens_details=None,
                    ),
                ),
            )

        monkeypatch.setattr(model, "stream_response", fake_stream_response)

        response = await model.get_response(
            system_instructions=None,
            input=[],
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )

        assert response.output == output
        assert response.usage.input_tokens == 3
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 8
        assert response.usage.input_tokens_details.cached_tokens == 0
        assert (
            getattr(response.usage.input_tokens_details, "cache_write_tokens", 0) == 0
        )
        assert response.usage.output_tokens_details.reasoning_tokens == 0

    async def test_upstream_runner_run_records_session_sample(
        self, rollout_context, monkeypatch
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisAgent,
            OsmosisLitellmModel,
            OsmosisMemorySession,
            OsmosisRolloutModel,
        )

        output = [
            ResponseOutputMessage(
                id="msg_1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="hello from rollout",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        async def fake_stream_response(self, *_args, **_kwargs):
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output=output, usage=None),
            )

        monkeypatch.setattr(
            OsmosisLitellmModel,
            "stream_response",
            fake_stream_response,
        )

        agent = OsmosisAgent(name="main", model=OsmosisRolloutModel())
        session = OsmosisMemorySession()

        result = await Runner.run(
            agent,
            "hello",
            session=session,
            run_config=RunConfig(tracing_disabled=True),
        )

        sample = await rollout_context.get_sample()
        assert result.final_output == "hello from rollout"
        assert sample is not None
        assert any(item.get("role") == "user" for item in sample.messages)
        assert any(item.get("role") == "assistant" for item in sample.messages)

    async def test_placeholder_model_direct_use_raises(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisRolloutModel,
        )

        model = OsmosisRolloutModel()

        with pytest.raises(NotImplementedError, match="placeholder"):
            await model.get_response()
