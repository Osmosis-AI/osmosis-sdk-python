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
    source = StrandsAgentSampleSource(
        SimpleNamespace(
            messages=messages,
            conversation_manager=SimpleNamespace(removed_message_count=0),
        )
    )

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


async def test_preview_does_not_claim_a_complete_history_after_strands_trims():
    from strands.agent.conversation_manager import SlidingWindowConversationManager

    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    manager = SlidingWindowConversationManager()
    agent = SimpleNamespace(
        messages=[
            {
                "role": "assistant" if index % 2 else "user",
                "content": [{"text": f"message {index}"}],
            }
            for index in range(50)
        ],
        conversation_manager=manager,
    )
    source = StrandsAgentSampleSource(agent)
    manager.apply_management(agent)

    preview = await source.get_preview(100)

    assert len(preview.messages) == 40
    assert preview.extra_fields == {
        "_preview_truncated": True,
        "_preview_turn": None,
    }


async def test_preview_formats_reasoning_and_media_without_warnings(caplog):
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )

    image = {"image": {"format": "png", "source": {"bytes": b"\x89PNG" * 4096}}}
    messages = [
        {"role": "user", "content": [{"text": "look"}, image]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                {"toolUse": {"toolUseId": "t1", "name": "shot", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [{"text": "captured"}, image],
                    }
                }
            ],
        },
    ]
    source = StrandsAgentSampleSource(
        SimpleNamespace(
            messages=messages,
            conversation_manager=SimpleNamespace(removed_message_count=0),
        )
    )

    caplog.set_level("WARNING")
    preview = await source.get_preview(10)

    assert not caplog.records
    assert preview is not None
    assert preview.messages == messages
    encoded = repr(preview.trajectory_messages)
    assert "[image omitted from preview]" in encoded
    assert "base64" not in encoded and "hmm" not in encoded
    assert preview.trajectory_messages[-2:] == [
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "captured\n[image omitted from preview]",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[image omitted from preview]"}],
        },
    ]


async def test_preview_step_ids_match_the_final_trajectory_with_media(tmp_path):
    from osmosis_ai.rollout.context import RolloutContext
    from osmosis_ai.rollout.integrations.agents.strands import (
        StrandsAgentSampleSource,
    )
    from osmosis_ai.rollout.trajectory import preview
    from osmosis_ai.rollout.trajectory.converter import convert_sample_to_trajectory

    image = {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}
    remote = {
        "image": {
            "format": "png",
            "source": {"location": {"type": "s3", "uri": "s3://bucket/a.png"}},
        }
    }
    messages = [
        {"role": "user", "content": [{"text": "go"}]},
        {"role": "user", "content": [remote]},
    ]
    for index in range(2):
        messages += [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                    {
                        "toolUse": {
                            "toolUseId": f"t{index}",
                            "name": "shot",
                            "input": {},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": f"t{index}",
                            "status": "success",
                            "content": [{"text": f"captured {index}"}, image],
                        }
                    }
                ],
            },
        ]
    messages.append({"role": "assistant", "content": [{"text": "done"}]})
    source = StrandsAgentSampleSource(
        SimpleNamespace(
            messages=messages,
            conversation_manager=SimpleNamespace(removed_message_count=0),
        )
    )
    path = tmp_path / "preview.json"

    await preview.capture_source(
        RolloutContext(rollout_id="r1", preview_path=path, sample_source=source)
    )

    snapshot = preview.read_snapshot(path)
    assert snapshot is not None
    steps = preview._preview_document(snapshot, "r1", None)["steps"]
    final = convert_sample_to_trajectory(await source.get_sample(), rollout_id="r1")
    assert [
        (step["extra"]["osmosis"]["original_step_id"], step["source"]) for step in steps
    ] == [(step.step_id, step.source) for step in final.steps]


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
    source = StrandsAgentSampleSource(
        SimpleNamespace(
            messages=messages,
            conversation_manager=SimpleNamespace(removed_message_count=0),
        )
    )
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
