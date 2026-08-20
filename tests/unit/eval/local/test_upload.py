"""Local-eval upload planning: strict validation and file selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.upload import (
    LocalEvalUploadError,
    _validate_upload_path,
    build_eval_upload_plan,
)

ROLLOUT_ID = "a" * 32
LOCAL_RUN_ID = "b" * 32


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _trajectory_payload(
    *,
    rollout_id: str = ROLLOUT_ID,
    row_index: int = 0,
    run_index: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": rollout_id,
        "agent": {"name": "echo", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": "hello"}],
        "extra": {
            "osmosis": {
                "rollout_id": rollout_id,
                "request_extra_fields": {
                    "row_index": row_index,
                    "run_index": run_index,
                },
            }
        },
    }


def _run_dir(tmp_path: Path, *, trajectory: bool = True) -> Path:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "local_run_id": LOCAL_RUN_ID,
            "run_name": "run-1",
            "created_at": "2026-08-18T01:00:00Z",
            "inputs": {
                "model_path": "openai/gpt-5-mini",
                "dataset": {"sha256": "c" * 64, "selected_source_rows": "0"},
                "n": 1,
                "rollout": {
                    "name": "echo",
                    "entrypoint": "main.py",
                    "source_digest": "d" * 64,
                },
                "versions": {
                    "rollout_protocol": "0.3",
                    "dataset_normalization": 1,
                    "state_schema": 1,
                },
            },
            "provenance": {
                "sdk_version": "0.3.0",
                "git_head": "e" * 40,
                "git_dirty": True,
                "config_branch": "feature/eval",
                "config_commit_sha": "f" * 40,
                "advanced": {"do_not_upload": True},
                "env": {"TOKEN": "secret"},
            },
        },
    )
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 1, "sampled_rows": 1, "total_dataset_rows": 1},
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "eval_run": {
                "id": LOCAL_RUN_ID,
                "name": "run-1",
                "status": "finished",
                "dataset_name": "dataset-1",
                "model_name": "openai/gpt-5-mini",
                "rollout_name": "echo",
                "started_at": "2026-08-18T01:00:00Z",
                "completed_at": "2026-08-18T01:01:00Z",
            },
            "summary": {"pass_threshold": 0.5, "passed": 1},
        },
    )
    row = {
        "row_index": 0,
        "run_index": 0,
        "rollout_id": ROLLOUT_ID,
        "status": "success",
    }
    if trajectory:
        row["trajectory_filename"] = "trajectory.json"
        _write_json(
            run_dir / "rollout_trials" / ROLLOUT_ID / "trajectory.json",
            _trajectory_payload(),
        )
    (run_dir / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    artifact = run_dir / "rollout_trials" / ROLLOUT_ID / "artifacts" / "answer.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"answer")
    _write_json(artifact.parent / "manifest.json", {"reserved": True})
    (run_dir / "logs.txt").write_bytes(b"log line\n")
    (run_dir / "events.jsonl").write_text("secret journal\n", encoding="utf-8")
    (run_dir / "summary.jsonl").write_text("projection\n", encoding="utf-8")
    return run_dir


def test_build_plan_selects_only_canonical_upload_files(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest_bytes = (run_dir / "manifest.json").read_bytes()

    plan = build_eval_upload_plan(run_dir)

    assert plan.manifest_digest == hashlib.sha256(manifest_bytes).hexdigest()
    assert plan.run == {
        "name": "run-1",
        "started_at": "2026-08-18T01:00:00Z",
        "completed_at": "2026-08-18T01:01:00Z",
        "experiment_config": {
            "rollout": {
                "name": "echo",
                "source_digest": "d" * 64,
                "branch": "feature/eval",
                "commit_sha": "f" * 40,
            },
            "entrypoint": "main.py",
            "model_path": "openai/gpt-5-mini",
            "dataset": {
                "name": "dataset-1",
                "sha256": "c" * 64,
                "selected_source_rows": "0",
            },
        },
        "evaluation_config": {"n": 1, "pass_threshold": 0.5},
    }
    assert plan.provenance == {
        "sdk_version": "0.3.0",
        "git_head": "e" * 40,
        "git_dirty": True,
        "config_branch": "feature/eval",
        "config_commit_sha": "f" * 40,
    }
    assert [file.path for file in plan.files] == [
        "index.jsonl",
        "progress.json",
        f"rollout_trials/{ROLLOUT_ID}/artifacts/answer.txt",
        f"rollout_trials/{ROLLOUT_ID}/trajectory.json",
    ]
    assert (run_dir / "logs.txt").is_file()
    assert all(
        file.sha256 == hashlib.sha256(file.source.read_bytes()).hexdigest()
        for file in plan.files
    )


def test_absent_trajectory_filename_is_valid_and_never_inferred(
    tmp_path: Path,
) -> None:
    plan = build_eval_upload_plan(_run_dir(tmp_path, trajectory=False))
    assert all(not file.path.endswith("trajectory.json") for file in plan.files)


def test_pending_run_is_rejected_by_index_progress_guard(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 2, "sampled_rows": 2, "total_dataset_rows": 2},
    )
    with pytest.raises(LocalEvalUploadError, match="incomplete"):
        build_eval_upload_plan(run_dir)


def test_sampled_rows_cannot_exceed_total_dataset_rows(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 1, "sampled_rows": 1, "total_dataset_rows": 0},
    )
    with pytest.raises(LocalEvalUploadError, match="exceeds total_dataset_rows"):
        build_eval_upload_plan(run_dir)


def test_total_runs_must_equal_sampled_rows_times_n(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 1, "sampled_rows": 2, "total_dataset_rows": 2},
    )
    with pytest.raises(LocalEvalUploadError, match="sampled_rows multiplied"):
        build_eval_upload_plan(run_dir)


def test_index_keys_must_cover_sampled_rows_and_n(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    row = json.loads((run_dir / "index.jsonl").read_text(encoding="utf-8"))
    row["row_index"] = 1
    (run_dir / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(LocalEvalUploadError, match="selected row range"):
        build_eval_upload_plan(run_dir)


def test_index_row_indices_can_map_to_noncontiguous_source_rows(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["dataset"]["selected_source_rows"] = "0,7"
    _write_json(manifest_path, manifest)
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 2, "sampled_rows": 2, "total_dataset_rows": 10},
    )
    rows = [
        {
            "row_index": 0,
            "run_index": 0,
            "rollout_id": ROLLOUT_ID,
            "status": "success",
        },
        {
            "row_index": 1,
            "run_index": 0,
            "rollout_id": "c" * 32,
            "status": "success",
        },
    ]
    (run_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    plan = build_eval_upload_plan(run_dir)
    assert plan.run["experiment_config"]["dataset"]["selected_source_rows"] == "0,7"


def test_index_row_indices_must_be_the_contiguous_selected_range(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["dataset"]["selected_source_rows"] = "0,7"
    _write_json(manifest_path, manifest)
    _write_json(
        run_dir / "progress.json",
        {"total_runs": 2, "sampled_rows": 2, "total_dataset_rows": 10},
    )
    rows = [
        {
            "row_index": 0,
            "run_index": 0,
            "rollout_id": ROLLOUT_ID,
            "status": "success",
        },
        {
            "row_index": 7,
            "run_index": 0,
            "rollout_id": "c" * 32,
            "status": "success",
        },
    ]
    (run_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(LocalEvalUploadError, match="selected row range"):
        build_eval_upload_plan(run_dir)


def test_oversized_integer_pass_threshold_is_a_validation_error(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["summary"]["pass_threshold"] = 10**400
    _write_json(metrics_path, metrics)
    with pytest.raises(LocalEvalUploadError, match="finite number"):
        build_eval_upload_plan(run_dir)


def test_trajectory_identity_must_match_index_rollout_id(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_json(
        run_dir / "rollout_trials" / ROLLOUT_ID / "trajectory.json",
        _trajectory_payload(rollout_id="c" * 32),
    )
    with pytest.raises(LocalEvalUploadError, match="does not identify"):
        build_eval_upload_plan(run_dir)


def test_trajectory_must_be_valid_atif(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    payload = _trajectory_payload()
    payload["steps"] = "not-a-step-list"
    _write_json(
        run_dir / "rollout_trials" / ROLLOUT_ID / "trajectory.json",
        payload,
    )
    with pytest.raises(LocalEvalUploadError, match="not valid ATIF"):
        build_eval_upload_plan(run_dir)


def test_trajectory_atif_validation_does_not_coerce_raw_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    payload = _trajectory_payload()
    payload["steps"][0]["step_id"] = "1"
    _write_json(
        run_dir / "rollout_trials" / ROLLOUT_ID / "trajectory.json",
        payload,
    )
    with pytest.raises(LocalEvalUploadError, match="not valid ATIF"):
        build_eval_upload_plan(run_dir)


def test_trajectory_row_run_identity_must_match_index(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    _write_json(
        run_dir / "rollout_trials" / ROLLOUT_ID / "trajectory.json",
        _trajectory_payload(row_index=9),
    )
    with pytest.raises(LocalEvalUploadError, match="row/run identity"):
        build_eval_upload_plan(run_dir)


def test_short_uppercase_config_commit_sha_is_preserved(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["config_commit_sha"] = "ABCDEF1"
    _write_json(manifest_path, manifest)

    plan = build_eval_upload_plan(run_dir)

    assert plan.provenance["config_commit_sha"] == "ABCDEF1"
    assert plan.run["experiment_config"]["rollout"]["commit_sha"] == "ABCDEF1"


def test_trajectory_filename_rejects_backslash(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, trajectory=False)
    row = {
        "row_index": 0,
        "run_index": 0,
        "rollout_id": ROLLOUT_ID,
        "status": "success",
        "trajectory_filename": r"trajectory\host.json",
    }
    (run_dir / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(LocalEvalUploadError, match="safe trajectory"):
        build_eval_upload_plan(run_dir)


def test_artifact_symlinks_fail_closed(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    artifact_root = run_dir / "rollout_trials" / ROLLOUT_ID / "artifacts"
    (artifact_root / "host.txt").symlink_to(run_dir / "manifest.json")
    with pytest.raises(LocalEvalUploadError, match="must not be a symlink"):
        build_eval_upload_plan(run_dir)


@pytest.mark.parametrize("name", ["bad\\name.txt", "bad\nname.txt"])
def test_artifact_files_with_unsafe_characters_fail_closed(
    tmp_path: Path, name: str
) -> None:
    run_dir = _run_dir(tmp_path)
    artifact_root = run_dir / "rollout_trials" / ROLLOUT_ID / "artifacts"
    (artifact_root / name).write_bytes(b"unsafe")
    with pytest.raises(LocalEvalUploadError, match="backslash or control"):
        build_eval_upload_plan(run_dir)


@pytest.mark.parametrize(
    "path",
    [
        "rollout_trials/a/artifacts/bad\\name.txt",
        "rollout_trials/a/artifacts/bad\nname.txt",
        "rollout_trials/a/artifacts//name.txt",
        "rollout_trials/a/artifacts/./name.txt",
        "rollout_trials/a/artifacts/../name.txt",
    ],
)
def test_artifact_upload_paths_reject_unsafe_characters_and_segments(
    path: str,
) -> None:
    with pytest.raises(LocalEvalUploadError):
        _validate_upload_path(path, where="artifact")


def test_artifact_upload_paths_reject_more_than_1024_characters() -> None:
    path = "rollout_trials/a/artifacts/" + "/".join(["x" * 200] * 5)
    assert len(path) > 1024
    with pytest.raises(LocalEvalUploadError, match="exceeds 1024"):
        _validate_upload_path(path, where="artifact")


def test_run_directory_symlink_is_rejected(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(run_dir, target_is_directory=True)
    with pytest.raises(LocalEvalUploadError, match="non-symlink run directory"):
        build_eval_upload_plan(alias)


def test_rollout_trials_root_symlink_is_rejected(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    real_trials = tmp_path / "linked-trials"
    trials_root = run_dir / "rollout_trials"
    trials_root.rename(real_trials)
    trials_root.symlink_to(real_trials, target_is_directory=True)
    with pytest.raises(
        LocalEvalUploadError, match="rollout_trials must be a regular, non-symlink"
    ):
        build_eval_upload_plan(run_dir)
