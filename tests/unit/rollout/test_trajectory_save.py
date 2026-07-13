"""Tests for osmosis_ai.rollout.trajectory.save."""

import json
from pathlib import Path
from typing import Any

from harbor.utils.trajectory_utils import format_trajectory_json

from osmosis_ai.rollout.trajectory import save_trajectories
from osmosis_ai.rollout.trajectory.report import (
    LlmCallMetrics,
    SampleReport,
    TrajectoryReport,
)
from osmosis_ai.rollout.types import ExecutionResult, RolloutSample, RolloutStatus


def make_result(**samples: RolloutSample) -> ExecutionResult:
    return ExecutionResult(status=RolloutStatus.SUCCESS, samples=samples)


def make_sample(sample_id: str, reward: float | None = None) -> RolloutSample:
    return RolloutSample(
        id=sample_id,
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
        result=make_result(s1=make_sample("s1", reward=1.0)),
        request_extra_fields={"eval_run_id": "er-1"},
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["schema_version"].startswith("ATIF-")
    assert doc["extra"]["osmosis"]["reward"] == 1.0
    assert doc["extra"]["osmosis"]["request_extra_fields"] == {"eval_run_id": "er-1"}
    assert artifact_file.read_text() == "hello"


async def test_multi_sample_rollout_writes_one_document_per_sample(
    tmp_path: Path,
) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=make_result(a=make_sample("a"), b=make_sample("b")),
        artifact_root=tmp_path,
    )

    assert (tmp_path / "r1" / "trajectory-a.json").exists()
    assert (tmp_path / "r1" / "trajectory-b.json").exists()
    assert not (tmp_path / "r1" / "trajectory.json").exists()


async def test_save_without_samples_writes_nothing(tmp_path: Path) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=ExecutionResult(status=RolloutStatus.FAILURE),
        artifact_root=tmp_path,
    )

    assert not (tmp_path / "r1").exists()


async def test_save_skips_sample_when_trajectory_conversion_was_unavailable(
    tmp_path: Path, caplog
) -> None:
    sample = RolloutSample(
        id="s1",
        messages=[{"role": "user", "content": "hi"}],
        trajectory_messages=None,
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(s1=sample),
        artifact_root=tmp_path,
    )

    assert not (tmp_path / "r1" / "trajectory.json").exists()
    assert any("conversion was unavailable" in r.message for r in caplog.records)


async def test_save_never_raises(tmp_path: Path) -> None:
    # A file where the rollout directory should be makes writes fail.
    (tmp_path / "r1").write_text("not a directory")

    await save_trajectories(
        rollout_id="r1",
        result=make_result(s1=make_sample("s1")),
        artifact_root=tmp_path,
    )


async def test_one_failed_document_does_not_block_later_samples(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    def failing_format(payload: dict[str, Any]) -> str:
        if payload["trajectory_id"] == "r1/a":
            raise ValueError("unserializable sample")
        return format_trajectory_json(payload)

    monkeypatch.setattr(
        "osmosis_ai.rollout.trajectory.save.format_trajectory_json", failing_format
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(a=make_sample("a"), b=make_sample("b")),
        artifact_root=tmp_path,
    )

    assert not (tmp_path / "r1" / "trajectory-a.json").exists()
    assert (tmp_path / "r1" / "trajectory-b.json").exists()
    assert any("Failed to write trajectory" in r.message for r in caplog.records)


async def test_sample_id_is_sanitized_for_filenames(tmp_path: Path) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=make_result(**{"a/b": make_sample("a/b"), "c": make_sample("c")}),
        artifact_root=tmp_path,
    )

    assert (tmp_path / "r1" / "trajectory-a_b.json").exists()
    assert (tmp_path / "r1" / "trajectory-c.json").exists()


async def test_report_is_dispatched_per_sample(tmp_path: Path) -> None:
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
        result=make_result(s1=make_sample("s1"), s2=make_sample("s2")),
        report=report,
        artifact_root=tmp_path,
    )

    s1 = json.loads((tmp_path / "r1" / "trajectory-s1.json").read_text())
    s2 = json.loads((tmp_path / "r1" / "trajectory-s2.json").read_text())
    assert s1["agent"]["model_name"] == "rollout-model"
    assert s1["final_metrics"]["total_prompt_tokens"] == 10
    assert s2["agent"]["model_name"] == "rollout-model"
    assert s2["final_metrics"] == {"total_steps": 2}


async def test_single_entry_report_matches_single_sample_regardless_of_key(
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
        result=make_result(s1=make_sample("s1")),
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["final_metrics"]["total_prompt_tokens"] == 10
    assert "unmatched_sample_reports" not in doc["extra"]["osmosis"]


async def test_unmatched_entries_are_preserved_for_single_sample_rollouts(
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
        result=make_result(s1=make_sample("s1")),
        report=report,
        artifact_root=tmp_path,
    )

    doc = json.loads((tmp_path / "r1" / "trajectory.json").read_text())
    assert doc["final_metrics"]["total_prompt_tokens"] == 10
    assert doc["extra"]["osmosis"]["unmatched_sample_reports"] == {
        "judge": {"llm_call_metrics": [{"prompt_tokens": 99}]}
    }
    assert any("unknown sample ids" in r.message for r in caplog.records)


async def test_unmatched_entries_are_dropped_for_multi_sample_rollouts(
    tmp_path: Path, caplog
) -> None:
    report = TrajectoryReport(
        samples={
            "nope": SampleReport(llm_call_metrics=[LlmCallMetrics(prompt_tokens=10)])
        }
    )

    await save_trajectories(
        rollout_id="r1",
        result=make_result(s1=make_sample("s1"), s2=make_sample("s2")),
        report=report,
        artifact_root=tmp_path,
    )

    for name in ("trajectory-s1.json", "trajectory-s2.json"):
        doc = json.loads((tmp_path / "r1" / name).read_text())
        assert doc["final_metrics"] == {"total_steps": 2}
        assert "unmatched_sample_reports" not in doc["extra"]["osmosis"]
    assert any("unknown sample ids" in r.message for r in caplog.records)
