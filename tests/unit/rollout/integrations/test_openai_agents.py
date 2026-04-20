"""Tests for osmosis_ai.rollout.integrations.agents.openai_agents."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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


class TestOsmosisOpenAIAgent:
    def test_requires_active_rollout_context(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        with pytest.raises(RuntimeError, match="active RolloutContext"):
            OsmosisOpenAIAgent(name="x")

    def test_builds_chat_completions_model_from_context(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        captured = {}

        def fake_client(*, base_url, api_key, **kwargs):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return MagicMock(name="AsyncOpenAI")

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI",
            side_effect=fake_client,
        ):
            agent = OsmosisOpenAIAgent(name="main")

        assert captured["base_url"] == "http://controller:9"
        assert captured["api_key"] == "test-key"
        assert agent.model is not None

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

    def test_uses_sentinel_api_key_when_missing(self):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        ctx = RolloutContext(
            chat_completions_url="http://controller:9",
            api_key=None,
            rollout_id="r1",
        )
        captured = {}

        def fake_client(*, base_url, api_key, **kwargs):
            captured["api_key"] = api_key
            return MagicMock(name="AsyncOpenAI")

        with (
            ctx,
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI",
                side_effect=fake_client,
            ),
        ):
            OsmosisOpenAIAgent(name="main")

        assert captured["api_key"] == "sk-osmosis-rollout"

    def test_user_supplied_model_is_respected(self, rollout_context):
        from agents.models.interface import Model

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        custom_model = MagicMock(spec=Model, name="UserModel")

        with patch(
            "osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"
        ) as mock_client:
            agent = OsmosisOpenAIAgent(name="main", model=custom_model)

        assert agent.model is custom_model
        mock_client.assert_not_called()


class _FakeRunResult:
    def __init__(self, messages):
        self._messages = messages

    def to_input_list(self):
        return self._messages


class TestOsmosisOpenAIAgentRun:
    async def test_injects_headers_via_context_var(self, rollout_context):
        from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE

        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        seen_headers = {}

        async def fake_run(agent, input, *, run_config, **kwargs):
            seen_headers.update(HEADERS_OVERRIDE.get() or {})
            return _FakeRunResult([{"role": "assistant", "content": "done"}])

        with (
            patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"),
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run",
                side_effect=fake_run,
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi")

        assert seen_headers["x-rollout-id"] == "rollout-xyz"
        assert seen_headers["x-sample-id"] == "main"

    async def test_records_sample_after_run(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        messages = [{"role": "assistant", "content": "42"}]

        async def fake_run(agent, input, *, run_config, **kwargs):
            return _FakeRunResult(messages)

        with (
            patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"),
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run",
                side_effect=fake_run,
            ),
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

        async def fake_run(agent, input, *, run_config, **kwargs):
            return _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with (
            patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"),
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run",
                side_effect=fake_run,
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

        async def fake_run(agent, input, *, run_config, **kwargs):
            return _FakeRunResult([{"role": "assistant", "content": "ok"}])

        with (
            patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"),
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run",
                side_effect=fake_run,
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi", sample_id="explicit-id")

        samples = rollout_context.get_samples()
        assert "explicit-id" in samples
        assert "main" not in samples

    async def test_raises_outside_rollout_context(self, rollout_context):
        from osmosis_ai.rollout.integrations.agents.openai_agents import (
            OsmosisOpenAIAgent,
        )

        # Construct inside the rollout context, then call run() after it exits.
        with patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"):
            agent = OsmosisOpenAIAgent(name="main")

        # Exit the fixture's rollout_context before calling run().
        from osmosis_ai.rollout.context import rollout_contextvar

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

        captured_run_config = {}

        async def fake_run(agent, input, *, run_config, **kwargs):
            captured_run_config["run_config"] = run_config
            return _FakeRunResult([{"role": "assistant", "content": "x"}])

        with (
            patch("osmosis_ai.rollout.integrations.agents.openai_agents.AsyncOpenAI"),
            patch(
                "osmosis_ai.rollout.integrations.agents.openai_agents.Runner.run",
                side_effect=fake_run,
            ),
        ):
            agent = OsmosisOpenAIAgent(name="main")
            await agent.run("hi")

        assert captured_run_config["run_config"].tracing_disabled is True
