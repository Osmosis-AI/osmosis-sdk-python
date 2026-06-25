from __future__ import annotations

from types import SimpleNamespace

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
        assert agent.model.model == "openai/osmosis-rollout"
        assert agent.model.base_url == "http://controller:9"
        assert agent.model.api_key == "test-key"

    async def test_rollout_model_merge_headers_succeeds_with_session(
        self, rollout_context
    ):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisLitellmModel,
            OsmosisMemorySession,
        )

        session = OsmosisMemorySession()
        await session.get_items()
        model = OsmosisLitellmModel()

        # Routing identity now lives in the chat_completions_url, so no per-call
        # rollout/sample headers are stamped. The merge must simply not raise
        # once a session (sample source) is registered with the context.
        headers = model._merge_headers(ModelSettings())

        assert isinstance(headers, dict)
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
