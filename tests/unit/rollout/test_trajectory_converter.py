"""Tests for osmosis_ai.rollout._trajectory.converter."""

from typing import Any

from osmosis_ai.rollout._trajectory.converter import (
    convert_sample_to_trajectory,
    messages_to_steps,
)
from osmosis_ai.rollout.types import RolloutSample


def make_sample(messages: list[dict[str, Any]], **kwargs: Any) -> RolloutSample:
    return RolloutSample(id="s1", messages=messages, **kwargs)


def test_basic_conversation_maps_to_sequential_steps() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        ),
        rollout_id="r1",
        sample_id="s1",
    )

    assert trajectory.session_id == "r1"
    assert trajectory.trajectory_id == "r1/s1"
    assert [s.source for s in trajectory.steps] == ["system", "user", "agent"]
    assert [s.step_id for s in trajectory.steps] == [1, 2, 3]
    assert trajectory.steps[2].message == "hello"


def test_tool_calls_and_results_fold_into_agent_step() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample(
            [
                {"role": "user", "content": "add 1+1"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 1, "b": 1}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "2"},
                {"role": "assistant", "content": "the answer is 2"},
            ]
        ),
        rollout_id="r1",
        sample_id="s1",
    )

    assert len(trajectory.steps) == 3
    agent_step = trajectory.steps[1]
    assert agent_step.tool_calls is not None
    assert agent_step.tool_calls[0].tool_call_id == "call_1"
    assert agent_step.tool_calls[0].function_name == "add"
    assert agent_step.tool_calls[0].arguments == {"a": 1, "b": 1}
    assert agent_step.observation is not None
    result = agent_step.observation.results[0]
    assert result.source_call_id == "call_1"
    assert result.content == "2"


def test_orphan_tool_result_keeps_original_id_in_extra() -> None:
    steps = messages_to_steps(
        [
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_other", "content": "out"},
        ]
    )

    result = steps[0].observation.results[0]
    assert result.source_call_id is None
    assert result.extra == {"tool_call_id": "call_other"}
    assert result.content == "out"


def test_tool_result_without_agent_step_becomes_system_observation() -> None:
    steps = messages_to_steps(
        [{"role": "tool", "tool_call_id": "c1", "content": "orphan output"}]
    )

    assert steps[0].source == "system"
    assert steps[0].observation.results[0].content == "orphan output"
    assert steps[0].observation.results[0].source_call_id is None


def test_unknown_role_is_kept_with_original_role() -> None:
    steps = messages_to_steps([{"role": "critic", "content": "needs work"}])

    assert steps[0].source == "user"
    assert steps[0].extra == {"original_role": "critic"}


def test_non_agent_step_demotes_agent_only_fields_to_extra() -> None:
    # ATIF forbids reasoning_content/tool_calls outside agent steps.
    steps = messages_to_steps(
        [
            {
                "role": "user",
                "content": "x",
                "reasoning_content": "internal",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
                ],
            }
        ]
    )

    assert steps[0].reasoning_content is None
    assert steps[0].tool_calls is None
    assert steps[0].extra is not None
    assert steps[0].extra["reasoning_content"] == "internal"
    assert steps[0].extra["tool_calls"][0]["tool_call_id"] == "c1"


def test_non_string_reasoning_content_is_serialized() -> None:
    steps = messages_to_steps(
        [
            {
                "role": "assistant",
                "content": "x",
                "reasoning_content": [{"type": "thinking", "text": "hmm"}],
            }
        ]
    )

    assert steps[0].reasoning_content is not None
    assert "hmm" in steps[0].reasoning_content


def test_content_part_flattening() -> None:
    steps = messages_to_steps(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "text", "text": "line two"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image_url", "image_url": {"url": "http://img"}},
                ],
            },
        ]
    )

    assert steps[0].message == "line one\nline two"
    assert "image_url" in steps[1].message


def test_malformed_tool_call_arguments_are_preserved_raw() -> None:
    steps = messages_to_steps(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c", "function": {"name": "f", "arguments": "not json"}}
                ],
            }
        ]
    )

    assert steps[0].tool_calls[0].arguments == {"_raw": "not json"}


def test_extra_carries_platform_context() -> None:
    sample = make_sample(
        [{"role": "user", "content": "x"}],
        label="ground truth",
        reward=0.5,
        metrics={"turns": 3},
        extra_fields={"custom": True},
    )
    trajectory = convert_sample_to_trajectory(
        sample,
        rollout_id="r1",
        sample_id="s1",
        request_label="request label",
        request_metadata={"dataset_row": {"q": "x"}},
        request_extra_fields={"eval_run_id": "er-1", "row_index": 3},
    )

    osmosis = trajectory.extra["osmosis"]
    assert osmosis["rollout_id"] == "r1"
    assert osmosis["sample_id"] == "s1"
    assert osmosis["label"] == "ground truth"
    assert osmosis["reward"] == 0.5
    assert osmosis["sample_metrics"] == {"turns": 3}
    assert osmosis["sample_extra_fields"] == {"custom": True}
    assert osmosis["request_metadata"] == {"dataset_row": {"q": "x"}}
    assert osmosis["request_extra_fields"] == {"eval_run_id": "er-1", "row_index": 3}


def test_request_label_used_when_sample_has_none() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample([{"role": "user", "content": "x"}]),
        rollout_id="r1",
        sample_id="s1",
        request_label="fallback",
    )

    assert trajectory.extra["osmosis"]["label"] == "fallback"
    assert "reward" not in trajectory.extra["osmosis"]


def test_trajectory_serializes_to_valid_atif_json() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample([{"role": "user", "content": "x"}], reward=1.0),
        rollout_id="r1",
        sample_id="s1",
    )
    doc = trajectory.to_json_dict()

    assert doc["schema_version"].startswith("ATIF-")
    assert doc["agent"]["name"] == "osmosis-rollout-sdk"
    assert doc["steps"][0]["step_id"] == 1
