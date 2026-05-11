"""Tests for osmosis_ai.rollout.context."""

from unittest.mock import MagicMock

import pytest

from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    HarborAgentWorkflowContext,
    RolloutContext,
    SampleSource,
    get_rollout_context,
    rollout_contextvar,
)
from osmosis_ai.rollout.types import AgentWorkflowConfig, RolloutSample


class StaticSampleSource(SampleSource):
    def __init__(self, messages):
        self.messages = messages

    async def get_sample(self, name: str) -> RolloutSample:
        return RolloutSample(id=name, messages=self.messages)


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
        """Direct contextvar set/reset preserves an outer rollout context."""
        outer = RolloutContext(chat_completions_url="http://outer", rollout_id="outer")
        inner = RolloutContext(chat_completions_url="http://inner", rollout_id="inner")

        with outer:
            assert get_rollout_context() is outer
            token = rollout_contextvar.set(inner)
            try:
                assert get_rollout_context() is inner
            finally:
                rollout_contextvar.reset(token)
            assert get_rollout_context() is outer

        assert get_rollout_context() is None

    async def test_register_sample_source_and_get_samples(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        messages = [{"role": "user", "content": "hi"}]

        ctx.register_sample_source("s1", StaticSampleSource(messages))
        samples = await ctx.get_samples()
        assert "s1" in samples
        assert isinstance(samples["s1"], RolloutSample)
        assert samples["s1"].id == "s1"
        assert samples["s1"].messages == messages

    async def test_get_samples_merges_sample_sources(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        first_messages = [{"role": "user", "content": [{"text": "hi"}]}]
        second_messages = [{"role": "assistant", "content": "done"}]

        ctx.register_sample_source("strands-sample", StaticSampleSource(first_messages))
        ctx.register_sample_source("openai-sample", StaticSampleSource(second_messages))

        samples = await ctx.get_samples()
        assert set(samples) == {"strands-sample", "openai-sample"}
        assert samples["strands-sample"].messages == first_messages
        assert samples["openai-sample"].messages == second_messages

    def test_register_sample_source_raises_on_duplicate(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        ctx.register_sample_source("s1", StaticSampleSource([]))
        with pytest.raises(ValueError, match="already exists"):
            ctx.register_sample_source("s1", StaticSampleSource([]))


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
