from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


async def test_sample_source_preserves_native_and_converts_messages() -> None:
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    sample = await StrandsAgentSampleSource(
        SimpleNamespace(messages=messages)
    ).get_sample()

    assert sample.messages == messages
    assert sample.trajectory_messages == [
        {"role": "user", "content": [{"text": "hello", "type": "text"}]}
    ]


async def test_sample_source_keeps_native_messages_when_conversion_fails(
    caplog,
) -> None:
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    with patch(
        "osmosis_ai.rollout.integrations.agents.strands.LiteLLMModel.format_request_messages",
        side_effect=RuntimeError("boom"),
    ):
        sample = await StrandsAgentSampleSource(
            SimpleNamespace(messages=messages)
        ).get_sample()

    assert sample.messages == messages
    assert sample.trajectory_messages is None
    assert any("Failed to convert Strands" in r.message for r in caplog.records)


class TestOsmosisStrandsAgentPromptConversion:
    def test_converts_openai_format(self):
        from osmosis_ai.rollout.integrations.agents.strands import (
            OsmosisStrandsAgent,
        )

        openai_messages = [
            {"role": "user", "content": "hello"},
        ]

        captured = {}

        def fake_init(self, *args, messages=None, **kwargs):
            captured["messages"] = messages

        with patch(
            "osmosis_ai.rollout.integrations.agents.strands.StrandsAgent.__init__",
            fake_init,
        ):
            OsmosisStrandsAgent(name="s", messages=openai_messages)

        assert captured["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]},
        ]

    def test_passes_none_messages_through(self):
        from osmosis_ai.rollout.integrations.agents.strands import (
            OsmosisStrandsAgent,
        )

        captured = {}

        def fake_init(self, *args, messages=None, **kwargs):
            captured["messages"] = messages

        with patch(
            "osmosis_ai.rollout.integrations.agents.strands.StrandsAgent.__init__",
            fake_init,
        ):
            OsmosisStrandsAgent(name="s")

        assert captured["messages"] is None
