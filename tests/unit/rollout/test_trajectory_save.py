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


async def test_native_atif_is_preserved_and_enriched_without_message_roundtrip(
    tmp_path: Path,
) -> None:
    native_document = {
        "schema_version": "ATIF-v1.7",
        "session_id": "harbor-session",
        "trajectory_id": "harbor-trajectory",
        "agent": {
            "name": "terminus-2",
            "version": "0.20.0",
            "model_name": "agent-reported-model",
            "extra": {"runtime": "native"},
        },
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "calling tool",
                "reasoning_content": "native reasoning",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "shell",
                        "arguments": {"command": "true"},
                    }
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call-1", "content": "native output"}
                    ]
                },
                "metrics": {"prompt_tokens": 999},
                "llm_call_count": 1,
                "extra": {"native-step": True},
            }
        ],
        "final_metrics": {"total_prompt_tokens": 999, "total_steps": 1},
        "extra": {"harbor": {"preserved": True}},
    }
    result = ExecutionResult(
        status=RolloutStatus.SUCCESS,
        sample=RolloutSample(
            label="expected",
            reward=0.75,
            trajectory_messages=None,
        ),
        trajectory_document=native_document,
    )
    report = TrajectoryReport(
        model_name="controller-model",
        samples={
            "single": SampleReport(
                llm_call_metrics=[LlmCallMetrics(prompt_tokens=11, completion_tokens=4)]
            )
        },
    )

    await save_trajectories(
        rollout_id="r1",
        result=result,
        request_metadata={"harbor_task": "org/task"},
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["session_id"] == "r1"
    assert doc["trajectory_id"] == "r1"
    assert doc["agent"]["model_name"] == "controller-model"
    assert doc["agent"]["extra"] == {"runtime": "native"}
    assert doc["steps"][0]["reasoning_content"] == "native reasoning"
    assert doc["steps"][0]["tool_calls"][0]["arguments"] == {"command": "true"}
    assert doc["steps"][0]["observation"]["results"][0]["content"] == ("native output")
    assert doc["steps"][0]["extra"] == {"native-step": True}
    assert doc["steps"][0]["metrics"]["prompt_tokens"] == 11
    assert doc["final_metrics"] == {
        "total_prompt_tokens": 11,
        "total_completion_tokens": 4,
        "total_steps": 1,
    }
    assert doc["extra"]["harbor"] == {"preserved": True}
    assert doc["extra"]["osmosis"]["native_session_id"] == "harbor-session"
    assert doc["extra"]["osmosis"]["native_trajectory_id"] == "harbor-trajectory"
    assert doc["extra"]["osmosis"]["reward"] == 0.75
    assert doc["extra"]["osmosis"]["request_metadata"] == {"harbor_task": "org/task"}
