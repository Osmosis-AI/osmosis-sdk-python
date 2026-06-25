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

    async def get_sample(self) -> RolloutSample:
        return RolloutSample(messages=self.messages)


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

    async def test_set_sample_source_and_get_sample(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        messages = [{"role": "user", "content": "hi"}]

        ctx.set_sample_source(StaticSampleSource(messages))
        sample = await ctx.get_sample()
        assert isinstance(sample, RolloutSample)
        assert sample.messages == messages

    async def test_get_sample_returns_none_without_source(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        assert await ctx.get_sample() is None

    def test_set_sample_source_raises_on_duplicate(self):
        ctx = RolloutContext(chat_completions_url="http://llm", rollout_id="r1")
        ctx.set_sample_source(StaticSampleSource([]))
        with pytest.raises(ValueError, match="already has a sample source"):
            ctx.set_sample_source(StaticSampleSource([]))


# ---------------------------------------------------------------------------
# GraderContext
# ---------------------------------------------------------------------------


class TestGraderContext:
    def test_sample_carried(self):
        sample = RolloutSample(messages=[])
        ctx = GraderContext(sample=sample)
        assert ctx.sample is sample

    def test_set_reward(self):
        sample = RolloutSample(messages=[])
        ctx = GraderContext(sample=sample)
        ctx.set_reward(0.8)
        assert ctx.sample.reward == 0.8

    def test_set_reward_missing_sample_raises(self):
        ctx = GraderContext(sample=None)
        with pytest.raises(ValueError, match="no sample to reward"):
            ctx.set_reward(1.0)

    def test_metadata_defaults_none(self):
        ctx = GraderContext()
        assert ctx.metadata is None

    def test_metadata_carried(self):
        metadata = {"tools": ["search"], "difficulty": 3}
        ctx = GraderContext(metadata=metadata)
        assert ctx.metadata == metadata

    def test_artifacts_default_none(self):
        ctx = GraderContext()
        assert ctx.artifacts is None

    def test_set_artifacts_stores_object(self):
        ctx = GraderContext()
        artifacts = {"judge": {"explanation": "ok"}}
        ctx.set_artifacts(artifacts)
        assert ctx.artifacts is artifacts

    def test_set_artifacts_replaces(self):
        ctx = GraderContext()
        ctx.set_artifacts({"a": 1})
        ctx.set_artifacts({"b": 2})
        assert ctx.artifacts == {"b": 2}

    def test_set_artifacts_rejects_non_dict(self):
        ctx = GraderContext()
        with pytest.raises(TypeError, match="artifacts must be a dict"):
            ctx.set_artifacts(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_input_metadata_and_output_artifacts_are_independent(self):
        """Reading input metadata and writing output artifacts cannot collide."""
        metadata = {"difficulty": 3}
        ctx = GraderContext(metadata=metadata)
        ctx.set_artifacts({"judge": {"explanation": "why"}})

        assert ctx.metadata == {"difficulty": 3}
        assert ctx.artifacts == {"judge": {"explanation": "why"}}
        # The write channel never mutates the read-only input metadata.
        assert "judge" not in ctx.metadata


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

    def test_metadata_defaults_none(self):
        ctx = AgentWorkflowContext(prompt=[{"role": "user", "content": "hi"}])
        assert ctx.metadata is None

    def test_metadata_carried(self):
        metadata = {"tools": ["search"]}
        ctx = AgentWorkflowContext(
            prompt=[{"role": "user", "content": "hi"}], metadata=metadata
        )
        assert ctx.metadata == metadata


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

    def test_positional_call_pattern_still_works(self):
        """Existing positional call sites (prompt, config, environment) keep working."""
        env = MagicMock()
        cfg = AgentWorkflowConfig(name="harbor-test")
        ctx = HarborAgentWorkflowContext(
            [{"role": "user", "content": "hi"}],
            cfg,
            env,
        )
        assert ctx.environment is env
        assert ctx.config is cfg
        assert ctx.metadata is None

    def test_metadata_threaded_through_super_init(self):
        env = MagicMock()
        cfg = AgentWorkflowConfig(name="harbor-test")
        metadata = {"tools": ["search"], "difficulty": 3}
        ctx = HarborAgentWorkflowContext(
            prompt=[{"role": "user", "content": "hi"}],
            config=cfg,
            environment=env,
            metadata=metadata,
        )
        assert ctx.metadata == metadata
        assert ctx.environment is env
