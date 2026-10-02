from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("limit", [-1, 0, 2, 10])
async def test_preview_is_bounded_and_isolated_from_grading_history(limit) -> None:
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    messages = [
        {"role": "user", "content": [{"text": text}]}
        for text in ("first", "second", "third")
    ]
    messages[0]["role"] = "assistant"
    source = StrandsAgentSampleSource(SimpleNamespace(messages=messages))

    preview = await source.get_preview(limit)

    if limit <= 0:
        assert preview is None
    else:
        assert preview is not None
        assert preview.messages == messages[-limit:]
        assert preview.trajectory_messages is not None
        assert preview.extra_fields == {
            "_preview_truncated": limit < 3,
            "_preview_turn": 1,
        }
        messages[-1]["content"][0]["text"] = "live update"
        assert preview.messages[-1]["content"][0]["text"] == "third"
        assert preview.trajectory_messages[-1]["content"][0]["text"] == "third"
        preview.messages[-1]["content"][0]["text"] = "preview edit"

    sample = await source.get_sample()
    assert len(sample.messages) == 3
    assert sample.messages == messages
    assert sample.extra_fields == {}
    assert sample.messages[-1]["content"][0]["text"] == (
        "live update" if limit > 0 else "third"
    )


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


@pytest.mark.parametrize("preview", [False, True])
async def test_sample_source_keeps_native_messages_when_conversion_fails(
    caplog,
    preview,
) -> None:
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    source = StrandsAgentSampleSource(SimpleNamespace(messages=messages))
    sensitive_text = "sensitive-preview-content"
    with patch(
        "osmosis_ai.rollout.integrations.agents.strands.LiteLLMModel.format_request_messages",
        side_effect=RuntimeError(sensitive_text),
    ):
        sample = await source.get_preview(10) if preview else await source.get_sample()

    assert sample is not None
    assert sample.messages == messages
    assert sample.trajectory_messages is None
    if preview:
        assert sensitive_text not in caplog.text
        assert not caplog.records
    else:
        assert any(
            "Failed to convert Strands" in r.message and r.exc_info is not None
            for r in caplog.records
        )


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
