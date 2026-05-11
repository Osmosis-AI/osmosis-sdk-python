import asyncio

import pytest

from osmosis_ai.eval.controller.state import ControllerRolloutState
from osmosis_ai.rollout.types import (
    GraderCompleteRequest,
    GraderStatus,
    RolloutCompleteRequest,
    RolloutSample,
    RolloutStatus,
)


def test_multi_sample_registers_created_sample_every_request_and_counts_completions() -> (
    None
):
    state = ControllerRolloutState(rollout_id="r1")

    state.register_chat_create_branch(
        "s1", "multi_sample", [{"role": "user", "content": "a"}], [{"name": "tool_a"}]
    )
    state.mark_chat_completion("s1", tokens=11)
    state.register_chat_create_branch(
        "s1", "multi_sample", [{"role": "user", "content": "b"}], [{"name": "tool_b"}]
    )
    state.mark_chat_completion("s1", tokens=7)

    assert state.controller_created_sample_ids == {"s1"}
    assert state.completion_counts == {"s1": 2}
    assert state.first_created_tools["s1"] == [{"name": "tool_a"}]
    assert state.current_request_tools_by_completion["s1"] == [
        [{"name": "tool_a"}],
        [{"name": "tool_b"}],
    ]
    assert state.total_tokens == 18


def test_single_sample_reuses_stored_tools_after_completion() -> None:
    state = ControllerRolloutState(rollout_id="r1")

    tools = state.register_chat_create_branch(
        "s1",
        mode="single_sample",
        messages=[{"role": "user", "content": "a"}],
        tools=[{"name": "first"}],
    )
    state.mark_chat_completion("s1", 1)
    reused = state.register_chat_reuse_branch(
        "s1",
        mode="single_sample",
        messages=[{"role": "user", "content": "b"}],
        tools=[{"name": "second"}],
    )

    assert tools == [{"name": "first"}]
    assert reused == [{"name": "first"}]
    assert state.first_created_tools["s1"] == [{"name": "first"}]


def test_single_sample_create_branch_reuses_stored_tools_after_completion() -> None:
    state = ControllerRolloutState(rollout_id="r1")

    state.register_chat_create_branch(
        "s1",
        mode="single_sample",
        messages=[{"role": "user", "content": "a"}],
        tools=[{"name": "first"}],
    )
    state.mark_chat_completion("s1", 1)
    reused = state.register_chat_create_branch(
        "s1",
        mode="single_sample",
        messages=[{"role": "user", "content": "b"}],
        tools=[{"name": "second"}],
    )

    assert reused == [{"name": "first"}]
    assert state.current_request_tools_by_completion["s1"] == [
        [{"name": "first"}],
        [{"name": "first"}],
    ]


def test_multi_sample_reuse_after_completion_uses_current_tools() -> None:
    state = ControllerRolloutState(rollout_id="r1")

    state.register_chat_create_branch(
        "s1", "multi_sample", [{"role": "user", "content": "a"}], [{"name": "first"}]
    )
    state.mark_chat_completion("s1", tokens=1)
    reused = state.register_chat_reuse_branch(
        "s1", "multi_sample", [{"role": "user", "content": "b"}], [{"name": "second"}]
    )

    assert reused == [{"name": "second"}]


def test_multi_sample_outcome_repeats_scored_ids_per_completion() -> None:
    state = ControllerRolloutState(rollout_id="r1")

    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_chat_completion("s1", tokens=1)
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_chat_completion("s1", tokens=1)
    state.register_chat_create_branch("s2", "multi_sample", [], None)
    state.mark_chat_completion("s2", tokens=1)
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={
                "s1": RolloutSample(id="s1", reward=1.0),
                "s2": RolloutSample(id="s2", reward=0.0),
            },
        )
    )

    outcome = state.to_outcome(duration_ms=1.0)

    assert outcome.scored_sample_ids == ["s1", "s1", "s2"]


def test_created_but_not_completed_single_sample_uses_current_tools_without_overwriting_first_tools() -> (
    None
):
    state = ControllerRolloutState(rollout_id="r1")

    state.register_chat_create_branch(
        "s1", "single_sample", [{"role": "user", "content": "a"}], [{"name": "first"}]
    )
    tools = state.register_chat_create_branch(
        "s1", "single_sample", [{"role": "user", "content": "b"}], [{"name": "second"}]
    )

    assert tools == [{"name": "second"}]
    assert state.first_created_tools["s1"] == [{"name": "first"}]
    assert state.completion_counts == {}


@pytest.mark.asyncio
async def test_callbacks_resolve_futures_and_status_is_diagnostic_only() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.register_chat_create_branch("s1", "multi_sample", [], None)

    state.mark_rollout_completed(
        RolloutCompleteRequest(
            rollout_id="r1",
            status=RolloutStatus.FAILURE,
            err_message="agent said failure",
        )
    )
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.FAILURE,
            err_message="grader said failure",
            samples={"s1": RolloutSample(id="s1", reward=1.0)},
        )
    )

    assert (await state.rollout_future) is not None
    assert (await state.grader_future).samples["s1"].reward == 1.0
    outcome = state.to_outcome(duration_ms=5.0)
    assert outcome.status is RolloutStatus.SUCCESS
    assert outcome.callback_diagnostics["rollout"]["status"] == "failure"
    assert outcome.callback_diagnostics["grader"]["status"] == "failure"


@pytest.mark.asyncio
async def test_duplicate_callbacks_are_stale_and_do_not_overwrite_outcome() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.register_chat_create_branch("s1", "multi_sample", [], None)

    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=1.0)},
        )
    )
    accepted = await state.grader_future
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=0.0)},
        )
    )

    outcome = state.to_outcome(duration_ms=5.0)

    assert accepted.samples["s1"].reward == 1.0
    assert outcome.samples["s1"].reward == 1.0
    assert outcome.callback_diagnostics["stale_callbacks"] == ["grader"]


@pytest.mark.asyncio
async def test_controller_error_resolves_rollout_future_and_marks_failure() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    rollout_future = state.rollout_future

    state.mark_controller_error("provider authentication failed")

    callback = await asyncio.wait_for(rollout_future, timeout=0.1)
    outcome = state.to_outcome(duration_ms=5.0)

    assert callback.status is RolloutStatus.FAILURE
    assert callback.err_message == "provider authentication failed"
    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error == "provider authentication failed"
    assert outcome.systemic_error == "provider authentication failed"


@pytest.mark.asyncio
async def test_controller_error_after_rollout_resolves_grader_future() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.mark_rollout_completed(
        RolloutCompleteRequest(rollout_id="r1", status=RolloutStatus.SUCCESS)
    )
    grader_future = state.grader_future

    state.mark_controller_error("provider down during grading wait")

    callback = await asyncio.wait_for(grader_future, timeout=0.1)
    outcome = state.to_outcome(duration_ms=5.0)

    assert callback.status is GraderStatus.FAILURE
    assert callback.samples == {}
    assert callback.err_message == "provider down during grading wait"
    assert outcome.status is RolloutStatus.FAILURE
    assert outcome.error == "provider down during grading wait"


def test_missing_unknown_none_rewards_and_empty_samples_fail() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    assert state.to_outcome(duration_ms=1.0).status is RolloutStatus.FAILURE
    assert "No samples returned" in state.to_outcome(duration_ms=1.0).error

    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=None)},
        )
    )
    assert "returned None as reward" in state.to_outcome(duration_ms=1.0).error

    state2 = ControllerRolloutState(rollout_id="r2")
    state2.register_chat_create_branch("s1", "multi_sample", [], None)
    state2.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r2",
            status=GraderStatus.SUCCESS,
            samples={"s2": RolloutSample(id="s2", reward=1.0)},
        )
    )
    assert "not created by the controller" in state2.to_outcome(duration_ms=1.0).error


def test_unknown_callback_sample_id_without_created_samples_reports_unknown() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s2": RolloutSample(id="s2", reward=1.0)},
        )
    )

    outcome = state.to_outcome(duration_ms=1.0)

    assert "not created by the controller" in outcome.error
    assert outcome.callback_diagnostics["grader"]["unknown_sample_ids"] == ["s2"]


def test_missing_callback_sample_id_is_reported_in_diagnostics() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={},
        )
    )

    outcome = state.to_outcome(duration_ms=1.0)

    assert "Missing grader rewards" in outcome.error
    assert outcome.callback_diagnostics["grader"]["missing_sample_ids"] == ["s1"]


def test_remove_sample_marks_skipped_after_reward_completeness_passes() -> None:
    state = ControllerRolloutState(rollout_id="r1")
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=1.0, remove_sample=True)},
        )
    )

    outcome = state.to_outcome(duration_ms=1.0)

    assert outcome.status is RolloutStatus.SUCCESS
    assert outcome.skipped is True
    assert outcome.scored_sample_ids == []
    assert outcome.skipped_sample_ids == ["s1"]


def test_skipped_sample_does_not_require_reward_and_is_reported_in_diagnostics() -> (
    None
):
    state = ControllerRolloutState(rollout_id="r1")
    state.register_chat_create_branch("s1", "multi_sample", [], None)
    state.mark_grader_completed(
        GraderCompleteRequest(
            rollout_id="r1",
            status=GraderStatus.SUCCESS,
            samples={"s1": RolloutSample(id="s1", reward=None, remove_sample=True)},
        )
    )

    outcome = state.to_outcome(1.0)

    assert outcome.status is RolloutStatus.SUCCESS
    assert outcome.scored_sample_ids == []
    assert outcome.skipped_sample_ids == ["s1"]
    assert outcome.callback_diagnostics["grader"]["available_sample_ids"] == ["s1"]
