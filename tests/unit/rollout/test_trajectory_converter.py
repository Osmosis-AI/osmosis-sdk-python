"""Tests for osmosis_ai.rollout.trajectory.converter."""

from typing import Any

from osmosis_ai.rollout.trajectory.converter import (
    convert_sample_to_trajectory,
    messages_to_steps,
)
from osmosis_ai.rollout.trajectory.report import LlmCall, SampleReport
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


def two_turn_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "done"},
    ]


def test_report_maps_calls_onto_agent_steps() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample(two_turn_messages()),
        rollout_id="r1",
        sample_id="s1",
        report=SampleReport(
            llm_calls=[
                LlmCall(
                    prompt_tokens=10,
                    completion_tokens=5,
                    logprobs=[-0.1],
                    prompt_token_ids=[101, 102],
                    completion_token_ids=[103],
                ),
                LlmCall(prompt_tokens=20, completion_tokens=7, model_name="m2"),
            ]
        ),
        default_model_name="m1",
    )

    assert trajectory.agent.model_name == "m1"
    agent_steps = [s for s in trajectory.steps if s.source == "agent"]
    assert agent_steps[0].metrics is not None
    assert agent_steps[0].metrics.prompt_tokens == 10
    assert agent_steps[0].metrics.logprobs == [-0.1]
    assert agent_steps[0].metrics.prompt_token_ids == [101, 102]
    assert agent_steps[0].metrics.completion_token_ids == [103]
    assert agent_steps[0].llm_call_count == 1
    assert agent_steps[1].model_name == "m2"
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 30
    assert trajectory.final_metrics.total_completion_tokens == 12


def test_mismatched_call_count_is_preserved_not_guessed() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample(two_turn_messages()),
        rollout_id="r1",
        sample_id="s1",
        report=SampleReport(llm_calls=[LlmCall(prompt_tokens=10)]),
    )

    assert all(s.metrics is None for s in trajectory.steps)
    assert trajectory.extra is not None
    assert trajectory.extra["osmosis"]["llm_calls"] == [{"prompt_tokens": 10}]
    # Totals still aggregate even without per-step attribution.
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 10


def test_explicit_final_metrics_win_over_summation() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample(two_turn_messages()),
        rollout_id="r1",
        sample_id="s1",
        report=SampleReport(
            llm_calls=[LlmCall(prompt_tokens=1), LlmCall(prompt_tokens=2)],
            final_metrics={"total_prompt_tokens": 99},
        ),
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 99


def test_sample_model_name_wins_over_rollout_default() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample([{"role": "user", "content": "x"}]),
        rollout_id="r1",
        sample_id="s1",
        report=SampleReport(model_name="sample-model"),
        default_model_name="rollout-model",
    )

    assert trajectory.agent.model_name == "sample-model"


def test_unmatched_sample_reports_are_preserved_in_extra() -> None:
    trajectory = convert_sample_to_trajectory(
        make_sample([{"role": "user", "content": "x"}]),
        rollout_id="r1",
        sample_id="s1",
        unmatched_sample_reports={
            "judge": SampleReport(llm_calls=[LlmCall(prompt_tokens=3)])
        },
    )

    assert trajectory.extra is not None
    assert trajectory.extra["osmosis"]["unmatched_sample_reports"] == {
        "judge": {"llm_calls": [{"prompt_tokens": 3}]}
    }


def test_inline_usage_top_level_shape() -> None:
    steps = messages_to_steps(
        [
            {
                "role": "assistant",
                "content": "hello",
                "model": "gpt-x",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        ]
    )

    assert steps[0].model_name == "gpt-x"
    assert steps[0].metrics is not None
    assert steps[0].metrics.prompt_tokens == 11
    assert steps[0].metrics.cached_tokens == 4
    assert steps[0].llm_call_count == 1


def test_inline_usage_harbor_extra_response_shape() -> None:
    # Harbor's extra.response convention, with Responses-API field names.
    steps = messages_to_steps(
        [
            {
                "role": "assistant",
                "content": "hello",
                "extra": {
                    "response": {
                        "model": "gpt-y",
                        "usage": {"input_tokens": 7, "output_tokens": 2},
                    }
                },
            }
        ]
    )

    assert steps[0].model_name == "gpt-y"
    assert steps[0].metrics is not None
    assert steps[0].metrics.prompt_tokens == 7
    assert steps[0].metrics.completion_tokens == 2


def test_inline_usage_ignored_on_non_agent_and_malformed() -> None:
    steps = messages_to_steps(
        [
            {"role": "user", "content": "x", "usage": {"prompt_tokens": 1}},
            {"role": "assistant", "content": "y", "usage": {"prompt_tokens": "NaN"}},
        ]
    )

    assert steps[0].metrics is None
    assert steps[1].metrics is None


def test_message_timestamps_are_normalized_to_iso() -> None:
    steps = messages_to_steps(
        [
            {"role": "user", "content": "x", "created_at": 1751000000},
            {"role": "assistant", "content": "y", "timestamp": "2026-07-08T12:00:00Z"},
            {"role": "user", "content": "z", "created_at": "not a date"},
        ]
    )

    assert steps[0].timestamp is not None
    assert steps[0].timestamp.startswith("2025-06-2")
    assert steps[1].timestamp == "2026-07-08T12:00:00+00:00"
    assert steps[2].timestamp is None
