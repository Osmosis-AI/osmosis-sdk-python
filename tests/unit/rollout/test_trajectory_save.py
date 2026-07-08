"""Tests for osmosis_ai.rollout._trajectory.save."""

import json
from pathlib import Path

from osmosis_ai.rollout._trajectory import save_trajectories
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


async def test_save_never_raises(tmp_path: Path) -> None:
    # A file where the rollout directory should be makes writes fail.
    (tmp_path / "r1").write_text("not a directory")

    await save_trajectories(
        rollout_id="r1",
        result=make_result(s1=make_sample("s1")),
        artifact_root=tmp_path,
    )


async def test_sample_id_is_sanitized_for_filenames(tmp_path: Path) -> None:
    await save_trajectories(
        rollout_id="r1",
        result=make_result(**{"a/b": make_sample("a/b"), "c": make_sample("c")}),
        artifact_root=tmp_path,
    )

    assert (tmp_path / "r1" / "trajectory-a_b.json").exists()
    assert (tmp_path / "r1" / "trajectory-c.json").exists()
