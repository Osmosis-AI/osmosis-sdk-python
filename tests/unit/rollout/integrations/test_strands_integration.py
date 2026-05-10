from __future__ import annotations

from unittest.mock import patch


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
