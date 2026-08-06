"""Tests for osmosis_ai.rollout.trajectory.save."""

import json
import logging
from pathlib import Path

from osmosis_ai.rollout.trajectory import save_trajectories
from osmosis_ai.rollout.trajectory.report import (
    LlmCallMetrics,
    SampleReport,
    TrajectoryReport,
)
from osmosis_ai.rollout.types import ExecutionResult, RolloutSample, RolloutStatus


def make_result(sample: RolloutSample | None = None) -> ExecutionResult:
    return ExecutionResult(status=RolloutStatus.SUCCESS, sample=sample)


def make_sample(reward: float | None = None) -> RolloutSample:
    return RolloutSample(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        reward=reward,
    )


async def test_save_writes_trajectory_next_to_artifacts(tmp_path: Path) -> None:
    artifact_file = tmp_path / "r1" / "artifacts" / "logs" / "out.txt"
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_text("hello")

    await save_trajectories(
        rollout_id="r1",
        result=make_result(make_sample(reward=1.0)),
        request_extra_fields={"eval_run_id": "er-1"},
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["schema_version"].startswith("ATIF-")
    assert doc["extra"]["osmosis"]["reward"] == 1.0
    assert doc["extra"]["osmosis"]["request_extra_fields"] == {"eval_run_id": "er-1"}
    assert artifact_file.read_text() == "hello"


async def test_save_without_sample_writes_nothing(tmp_path: Path) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=ExecutionResult(status=RolloutStatus.FAILURE),
        artifact_root=tmp_path,
    )

    assert not (tmp_path / "r1").exists()


async def test_save_skips_sample_without_trajectory_messages(
    tmp_path: Path, caplog
) -> None:
    # Explicit None is the documented way to disable persistence, and upstream
    # conversion failures already warn at their source: skipping must stay quiet
    # at warning level while still recording that the transcript was dropped.
    caplog.set_level(logging.INFO, logger="osmosis_ai.rollout.trajectory.save")
    sample = RolloutSample(
        messages=[{"role": "user", "content": "hi"}],
        trajectory_messages=None,
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(sample),
        artifact_root=tmp_path,
    )

    assert not (tmp_path / "r1" / "trajectory.json").exists()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Skipping trajectory for rollout r1" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    )


async def test_save_never_raises(tmp_path: Path) -> None:
    # A file where the rollout directory should be makes writes fail.
    (tmp_path / "r1").write_text("not a directory")

    await save_trajectories(
        rollout_id="r1",
        result=make_result(make_sample()),
        artifact_root=tmp_path,
    )


async def test_sample_less_failure_writes_diagnostics_sidecar(tmp_path: Path) -> None:
    """A failed rollout with no sample must still leave a durable record."""
    await save_trajectories(
        rollout_id="r1",
        result=ExecutionResult(
            status=RolloutStatus.FAILURE,
            extra_fields={"backend": "harbor-v2", "phase": "setup"},
        ),
        artifact_root=tmp_path,
    )

    sidecar = json.loads((tmp_path / "r1" / "diagnostics.json").read_text())
    assert sidecar["phase"] == "setup"
    assert not (tmp_path / "r1" / "trajectory.json").exists()


async def test_diagnostics_override_wins_over_result_extra_fields(
    tmp_path: Path,
) -> None:
    """The retained latest diagnostics may come from a different result than
    the archived sample; the explicit override must win."""
    await save_trajectories(
        rollout_id="r1",
        result=ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=make_sample(reward=1.0),
            extra_fields={"phase": "agent"},
        ),
        diagnostics={"phase": "grading"},
        artifact_root=tmp_path,
    )

    sidecar = json.loads((tmp_path / "r1" / "diagnostics.json").read_text())
    assert sidecar["phase"] == "grading"
    # The trajectory document still carries the archived result's own fields.
    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["extra"]["osmosis"]["result_extra_fields"] == {"phase": "agent"}


async def test_result_extra_fields_land_in_trajectory_extra(tmp_path: Path) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=ExecutionResult(
            status=RolloutStatus.SUCCESS,
            sample=make_sample(reward=0.5),
            extra_fields={"backend": "harbor-v2", "timings_sec": {"agent": 3.0}},
        ),
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["extra"]["osmosis"]["result_extra_fields"]["backend"] == "harbor-v2"
    assert (tmp_path / "r1" / "diagnostics.json").exists()


async def test_report_metrics_land_in_document(tmp_path: Path) -> None:
    report = TrajectoryReport(
        model_name="rollout-model",
        samples={
            "s1": SampleReport(
                llm_call_metrics=[LlmCallMetrics(prompt_tokens=10, completion_tokens=5)]
            )
        },
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(make_sample()),
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["agent"]["model_name"] == "rollout-model"
    assert doc["final_metrics"]["total_prompt_tokens"] == 10


async def test_single_entry_report_matches_sample_regardless_of_key(
    tmp_path: Path,
) -> None:
    report = TrajectoryReport(
        samples={
            "whatever": SampleReport(
                llm_call_metrics=[LlmCallMetrics(prompt_tokens=10)]
            )
        }
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(make_sample()),
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["final_metrics"]["total_prompt_tokens"] == 10
    assert "unmatched_sample_reports" not in doc["extra"]["osmosis"]


async def test_multi_entry_report_is_preserved_not_guessed(
    tmp_path: Path, caplog
) -> None:
    report = TrajectoryReport(
        samples={
            "s1": SampleReport(llm_call_metrics=[LlmCallMetrics(prompt_tokens=10)]),
            "judge": SampleReport(llm_call_metrics=[LlmCallMetrics(prompt_tokens=99)]),
        }
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(make_sample()),
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert "total_prompt_tokens" not in doc["final_metrics"]
    assert doc["extra"]["osmosis"]["unmatched_sample_reports"] == {
        "s1": {"llm_call_metrics": [{"prompt_tokens": 10}]},
        "judge": {"llm_call_metrics": [{"prompt_tokens": 99}]},
    }
    assert any("preserving them under" in r.getMessage() for r in caplog.records)


def native_document() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "harbor-session-9",
        "trajectory_id": "harbor-traj-9",
        "agent": {
            "name": "terminus-2",
            "version": "1.0",
            "model_name": "openai/native",
            "tool_definitions": [{"name": "bash"}],
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "fix"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "patching",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash",
                        "arguments": {"cmd": "pytest"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "c1", "content": "ok"}]},
                "metrics": {
                    "prompt_tokens": 10,
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [7],
                    "logprobs": [-0.5],
                },
            },
        ],
    }


async def test_native_document_archived_without_reconstruction(
    tmp_path: Path,
) -> None:
    """The Harbor-authored ATIF keeps its step structure, tool calls, token
    ids and logprobs; Osmosis context is overlaid under extra.osmosis."""
    sample = RolloutSample(
        messages=[{"role": "assistant", "content": "patching"}],
        trajectory_messages=None,
        label="row-3",
        reward=1.0,
        metrics={"input_tokens": 10, "output_tokens": 4, "cost_usd": 0.01},
    )
    result = ExecutionResult(
        status=RolloutStatus.SUCCESS,
        sample=sample,
        trajectory_document=native_document(),
        extra_fields={"backend": "harbor-v2", "phase": "verifier"},
    )

    await save_trajectories(rollout_id="r1", result=result, artifact_root=tmp_path)

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    agent_step = doc["steps"][1]
    assert agent_step["tool_calls"][0]["function_name"] == "bash"
    assert agent_step["observation"]["results"][0]["content"] == "ok"
    assert agent_step["metrics"]["prompt_token_ids"] == [1, 2, 3]
    assert agent_step["metrics"]["logprobs"] == [-0.5]
    assert doc["agent"]["name"] == "terminus-2"
    assert doc["agent"]["tool_definitions"] == [{"name": "bash"}]
    # Ids are normalized for platform joins; harbor's own are preserved.
    assert doc["session_id"] == "r1"
    assert doc["extra"]["osmosis"]["native_session_id"] == "harbor-session-9"
    assert doc["extra"]["osmosis"]["native_trajectory_id"] == "harbor-traj-9"
    assert doc["extra"]["osmosis"]["reward"] == 1.0
    assert doc["extra"]["osmosis"]["label"] == "row-3"
    assert doc["extra"]["osmosis"]["result_extra_fields"]["phase"] == "verifier"


async def test_native_final_metrics_seeded_from_trial_totals(
    tmp_path: Path,
) -> None:
    """Harbor's per-trial token accounting seeds FinalMetrics when the
    document carries none; total_steps is always recomputed."""
    sample = RolloutSample(
        messages=[],
        trajectory_messages=None,
        metrics={"input_tokens": 10, "output_tokens": 4, "cost_usd": 0.01},
    )
    result = ExecutionResult(
        status=RolloutStatus.SUCCESS,
        sample=sample,
        trajectory_document=native_document(),
    )

    await save_trajectories(rollout_id="r1", result=result, artifact_root=tmp_path)

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["final_metrics"]["total_prompt_tokens"] == 10
    assert doc["final_metrics"]["total_completion_tokens"] == 4
    assert doc["final_metrics"]["total_cost_usd"] == 0.01
    assert doc["final_metrics"]["total_steps"] == 2


async def test_native_controller_report_overlays_step_metrics(
    tmp_path: Path,
) -> None:
    """Controller-reported metrics win field by field without touching the
    document's step structure or identifiers."""
    report = TrajectoryReport(
        model_name="openai/controller-view",
        samples={
            "s1": SampleReport(
                llm_call_metrics=[
                    LlmCallMetrics(prompt_tokens=99, completion_tokens=42)
                ],
                final_metrics={"total_prompt_tokens": 99},
            )
        },
    )
    result = ExecutionResult(
        status=RolloutStatus.SUCCESS,
        sample=RolloutSample(messages=[], trajectory_messages=None),
        trajectory_document=native_document(),
    )

    await save_trajectories(
        rollout_id="r1", result=result, report=report, artifact_root=tmp_path
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    agent_step = doc["steps"][1]
    assert agent_step["metrics"]["prompt_tokens"] == 99
    assert agent_step["metrics"]["completion_tokens"] == 42
    # Native retokenization fields survive the count-only overlay.
    assert agent_step["metrics"]["prompt_token_ids"] == [1, 2, 3]
    assert agent_step["metrics"]["completion_token_ids"] == [7]
    assert agent_step["metrics"]["logprobs"] == [-0.5]
    assert doc["final_metrics"]["total_prompt_tokens"] == 99
    assert doc["agent"]["model_name"] == "openai/controller-view"
    # Structure untouched: same steps, same tool calls.
    assert len(doc["steps"]) == 2
    assert agent_step["tool_calls"][0]["tool_call_id"] == "c1"


async def test_native_document_wins_over_converter_path(tmp_path: Path) -> None:
    """When a document is present the converter must not run at all, even for
    a sample that would otherwise produce a rebuilt trajectory."""
    result = ExecutionResult(
        status=RolloutStatus.SUCCESS,
        sample=make_sample(reward=1.0),  # has default trajectory_messages
        trajectory_document=native_document(),
    )

    await save_trajectories(rollout_id="r1", result=result, artifact_root=tmp_path)

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    # The converter would have produced osmosis-rollout-sdk as the agent.
    assert doc["agent"]["name"] == "terminus-2"
