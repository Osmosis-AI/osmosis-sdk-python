"""Contract tests for the SDK-owned ATIF v1.7 models and serializer."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from osmosis_ai.rollout.trajectory._atif import (
    Agent,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
    format_trajectory_json,
)


def _trajectory(*steps: Step, **kwargs: Any) -> Trajectory:
    return Trajectory(
        agent=Agent(name="osmosis-rollout-sdk", version="0.3.0"),
        steps=list(steps) or [Step(step_id=1, source="user", message="hello")],
        **kwargs,
    )


def test_models_keep_the_atif_v17_wire_fields() -> None:
    expected_fields: dict[type[BaseModel], list[str]] = {
        Agent: ["name", "version", "model_name", "tool_definitions", "extra"],
        ImageSource: ["media_type", "path"],
        ContentPart: ["type", "text", "source"],
        Metrics: [
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "cost_usd",
            "prompt_token_ids",
            "completion_token_ids",
            "logprobs",
            "extra",
        ],
        FinalMetrics: [
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_cached_tokens",
            "total_cost_usd",
            "total_steps",
            "extra",
        ],
        ToolCall: ["tool_call_id", "function_name", "arguments", "extra"],
        SubagentTrajectoryRef: [
            "trajectory_id",
            "session_id",
            "trajectory_path",
            "extra",
        ],
        ObservationResult: [
            "source_call_id",
            "content",
            "subagent_trajectory_ref",
            "extra",
        ],
        Observation: ["results"],
        Step: [
            "step_id",
            "timestamp",
            "source",
            "model_name",
            "reasoning_effort",
            "message",
            "reasoning_content",
            "tool_calls",
            "observation",
            "metrics",
            "is_copied_context",
            "llm_call_count",
            "extra",
        ],
        Trajectory: [
            "schema_version",
            "session_id",
            "trajectory_id",
            "agent",
            "steps",
            "notes",
            "final_metrics",
            "continued_trajectory_ref",
            "extra",
            "subagent_trajectories",
        ],
    }

    for model, fields in expected_fields.items():
        assert list(model.model_fields) == fields
        assert model.model_config["extra"] == "forbid"


def test_trajectory_serializes_to_the_existing_json_shape() -> None:
    trajectory = _trajectory(
        Step(step_id=1, source="user", message="calculate"),
        Step(
            step_id=2,
            timestamp="2026-07-29T08:00:00Z",
            source="agent",
            model_name="model-a",
            reasoning_effort="high",
            message="",
            reasoning_content="thinking",
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    function_name="add",
                    arguments={"a": 1, "b": 2},
                )
            ],
            observation=Observation(
                results=[ObservationResult(source_call_id="call-1", content="3")]
            ),
            metrics=Metrics(
                prompt_tokens=2,
                completion_tokens=1,
                prompt_token_ids=[10, 11],
                logprobs=[-0.25],
            ),
            is_copied_context=False,
            llm_call_count=1,
        ),
        session_id="session-1",
        trajectory_id="trajectory-1",
        final_metrics=FinalMetrics(total_prompt_tokens=2, total_steps=2),
        extra={"osmosis": {"reward": None}},
    )

    assert trajectory.to_json_dict() == {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "trajectory_id": "trajectory-1",
        "agent": {"name": "osmosis-rollout-sdk", "version": "0.3.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "calculate"},
            {
                "step_id": 2,
                "timestamp": "2026-07-29T08:00:00Z",
                "source": "agent",
                "model_name": "model-a",
                "reasoning_effort": "high",
                "message": "",
                "reasoning_content": "thinking",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "add",
                        "arguments": {"a": 1, "b": 2},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call-1", "content": "3"}]
                },
                "metrics": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "prompt_token_ids": [10, 11],
                    "logprobs": [-0.25],
                },
                "is_copied_context": False,
                "llm_call_count": 1,
            },
        ],
        "final_metrics": {"total_prompt_tokens": 2, "total_steps": 2},
        # None model fields are omitted, but arbitrary metadata is preserved.
        "extra": {"osmosis": {"reward": None}},
    }


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            {"step_id": 0, "source": "user", "message": "x"},
            "greater than or equal to 1",
        ),
        (
            {
                "step_id": 1,
                "source": "user",
                "message": "x",
                "metrics": {"prompt_tokens": 1},
            },
            "only applicable when source is 'agent'",
        ),
        (
            {
                "step_id": 1,
                "source": "agent",
                "message": "x",
                "reasoning_content": "thinking",
                "llm_call_count": 0,
            },
            "must be absent when llm_call_count is 0",
        ),
        (
            {
                "step_id": 1,
                "source": "user",
                "message": "x",
                "timestamp": "not-a-timestamp",
            },
            "Invalid ISO 8601 timestamp",
        ),
    ],
)
def test_step_validation(step: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Step.model_validate(step)


def test_trajectory_validates_step_order_and_tool_references() -> None:
    with pytest.raises(ValidationError, match=r"expected 1 .* got 2"):
        _trajectory(Step(step_id=2, source="user", message="x"))

    bad_reference = Step(
        step_id=1,
        source="agent",
        message="x",
        observation=Observation(
            results=[ObservationResult(source_call_id="missing", content="result")]
        ),
    )
    with pytest.raises(ValidationError, match="not found in step 1's tool_calls"):
        _trajectory(bad_reference)


def test_multimodal_and_subagent_contract_validation() -> None:
    with pytest.raises(ValidationError, match="required when type='image'"):
        ContentPart(type="image")
    with pytest.raises(ValidationError, match="must be resolvable"):
        SubagentTrajectoryRef(session_id="informational-only")

    image = ContentPart(
        type="image",
        source=ImageSource(media_type="image/png", path="image.png"),
    )
    assert _trajectory(
        Step(step_id=1, source="user", message=[image])
    ).has_multimodal_content()

    subagent = _trajectory(trajectory_id="subagent-1")
    with pytest.raises(ValidationError, match="not unique"):
        _trajectory(subagent_trajectories=[subagent, subagent.model_copy(deep=True)])


def test_formatter_pretty_prints_with_compact_numeric_arrays() -> None:
    formatted = format_trajectory_json(
        {
            "token_ids": [1, 2, -3],
            "logprobs": [-0.1, 2e-5],
            "mixed": [1, None],
            "nested": [[1, 2], [3, 4]],
            "message": "café",
        }
    )

    assert (
        formatted
        == """{
  \"token_ids\": [1, 2, -3],
  \"logprobs\": [-0.1, 2e-05],
  \"mixed\": [
    1,
    null
  ],
  \"nested\": [
    [1, 2],
    [3, 4]
  ],
  \"message\": \"caf\\u00e9\"
}"""
    )
