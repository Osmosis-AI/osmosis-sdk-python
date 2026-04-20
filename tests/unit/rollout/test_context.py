"""Tests for osmosis_ai.rollout.context."""

from unittest.mock import MagicMock

import pytest

from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    HarborAgentWorkflowContext,
    RolloutContext,
    get_rollout_context,
    rollout_contextvar,
)
from osmosis_ai.rollout.types import AgentWorkflowConfig, RolloutSample

# ---------------------------------------------------------------------------
# RolloutContext
# ---------------------------------------------------------------------------


class TestRolloutContext:
    def test_defaults(self):
        ctx = RolloutContext(
            chat_completions_url="http://llm",
            api_key="key",
            rollout_id="r1",
        )
        assert ctx.chat_completions_url == "http://llm"
        assert ctx.api_key == "key"
        assert ctx.rollout_id == "r1"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("OSMOSIS_CHAT_COMPLETIONS_URL", "http://env-llm")
        monkeypatch.setenv("OSMOSIS_API_KEY", "env-key")
        monkeypatch.setenv("OSMOSIS_ROLLOUT_ID", "env-r1")

        ctx = RolloutContext()
        assert ctx.chat_completions_url == "http://env-llm"
        assert ctx.api_key == "env-key"
        assert ctx.rollout_id == "env-r1"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OSMOSIS_CHAT_COMPLETIONS_URL", "http://env")
        ctx = RolloutContext(chat_completions_url="http://explicit")
        assert ctx.chat_completions_url == "http://explicit"

    def test_context_manager(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        assert get_rollout_context() is None

        with ctx:
            assert get_rollout_context() is ctx
        assert get_rollout_context() is None

    def test_contextvar_reset_restores_outer(self):
        """InProcessDriver pattern: set/reset on contextvar preserves outer value."""
        outer = RolloutContext(chat_completions_url="http://outer", rollout_id="outer")
        inner = RolloutContext(chat_completions_url="http://inner", rollout_id="inner")

        with outer:
            assert get_rollout_context() is outer
            # InProcessDriver uses contextvar.set/reset directly (not with)
            token = rollout_contextvar.set(inner)
            try:
                assert get_rollout_context() is inner
            finally:
                rollout_contextvar.reset(token)
            assert get_rollout_context() is outer

        assert get_rollout_context() is None

    def test_register_agent_and_get_samples(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        agent = MagicMock()
        agent.messages = [{"role": "user", "content": "hi"}]

        ctx.register_agent("s1", agent)
        samples = ctx.get_samples()
        assert "s1" in samples
        assert isinstance(samples["s1"], RolloutSample)
        assert samples["s1"].id == "s1"
        assert samples["s1"].messages == agent.messages

    def test_record_sample_and_get_samples(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        messages = [{"role": "assistant", "content": "done"}]
        ctx.record_sample("s1", messages)

        samples = ctx.get_samples()
        assert "s1" in samples
        assert isinstance(samples["s1"], RolloutSample)
        assert samples["s1"].id == "s1"
        assert samples["s1"].messages == messages

    def test_get_samples_merges_both_tracks(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")

        strands_agent = MagicMock()
        strands_agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        ctx.register_agent("strands-sample", strands_agent)

        oa_messages = [{"role": "assistant", "content": "done"}]
        ctx.record_sample("openai-sample", oa_messages)

        samples = ctx.get_samples()
        assert set(samples) == {"strands-sample", "openai-sample"}
        assert samples["strands-sample"].messages == strands_agent.messages
        assert samples["openai-sample"].messages == oa_messages

    def test_record_sample_raises_on_duplicate(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        ctx.record_sample("s1", [])
        with pytest.raises(ValueError, match="already used"):
            ctx.record_sample("s1", [])

    def test_record_sample_raises_on_register_agent_collision(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        agent = MagicMock()
        agent.messages = []
        ctx.register_agent("s1", agent)
        with pytest.raises(ValueError, match="already used"):
            ctx.record_sample("s1", [])


# ---------------------------------------------------------------------------
# GraderContext
# ---------------------------------------------------------------------------


class TestGraderContext:
    def test_get_samples(self):
        sample = RolloutSample(id="s1", messages=[])
        ctx = GraderContext(samples={"s1": sample})
        assert ctx.get_samples() == {"s1": sample}

    def test_set_sample_reward(self):
        sample = RolloutSample(id="s1", messages=[])
        ctx = GraderContext(samples={"s1": sample})
        ctx.set_sample_reward("s1", 0.8)
        assert ctx.samples["s1"].reward == 0.8

    def test_set_reward_missing_sample_raises(self):
        ctx = GraderContext(samples={})
        with pytest.raises(ValueError, match="Sample unknown not found"):
            ctx.set_sample_reward("unknown", 1.0)


# ---------------------------------------------------------------------------
# AgentWorkflowContext / HarborAgentWorkflowContext
# ---------------------------------------------------------------------------


class TestAgentWorkflowContext:
    def test_basic_construction(self):
        prompt = [{"role": "user", "content": "hello"}]
        ctx = AgentWorkflowContext(prompt=prompt)
        assert ctx.prompt == prompt
        assert ctx.config is None

    def test_with_config(self):
        cfg = AgentWorkflowConfig(name="test")
        ctx = AgentWorkflowContext(
            prompt=[{"role": "user", "content": "hi"}], config=cfg
        )
        assert ctx.config is cfg


class TestHarborAgentWorkflowContext:
    def test_with_environment(self):
        env = MagicMock()
        cfg = AgentWorkflowConfig(name="harbor-test")
        ctx = HarborAgentWorkflowContext(
            prompt=[{"role": "user", "content": "hi"}],
            config=cfg,
            environment=env,
        )
        assert ctx.environment is env
        assert ctx.config is cfg
