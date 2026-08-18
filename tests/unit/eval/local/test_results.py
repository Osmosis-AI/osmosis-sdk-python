"""Materializer tests: index rows, metrics formula, and projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.local.results import (
    Materializer,
    RunIdentity,
    aggregate_metrics,
    atif_rollout_identity,
    build_index_row,
    pass_at_k,
    read_valid_trajectory,
    render_index_lines,
    safe_artifact_relative_paths,
    select_attempts,
)
from osmosis_ai.eval.local.state import TerminalRecord

ROLLOUT_A = "a" * 32
ROLLOUT_B = "b" * 32


def _record(
    row: int, run: int, *, rollout_id: str | None = None, **overrides: Any
) -> TerminalRecord:
    payload: dict[str, Any] = {
        "row_index": row,
        "run_index": run,
        "rollout_id": rollout_id or f"{row:016x}{run:016x}",
        "status": "success",
        "reward": 1.0,
        "tokens": 11,
        "duration_ms": 42.0,
    }
    payload.update(overrides)
    return TerminalRecord(**payload)


def _atif(rollout_id: str, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "session_id": rollout_id,
        "trajectory_id": rollout_id,
        "steps": [],
        "extra": {"osmosis": {"rollout_id": rollout_id, "reward": 1.0}},
    }
    document.update(overrides)
    return document


def _write_trajectory(trials_dir: Path, rollout_id: str, document: Any) -> Path:
    path = trials_dir / rollout_id / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# index.jsonl rows (§2.2)
# --------------------------------------------------------------------------- #

#: The monolith eval controller's index schema -- counterparty:
#: ``iac/aws-eks-workloads/cloud-eval/eval-controller/run_eval.py``
#: ``INDEX_FIELDS``. Parity is a documented intent, not a golden-fixture
#: contract. Individual rows omit their ``None`` fields, so the claim below is
#: about the full key set ``build_index_row`` can emit, not about any one row.
MONOLITH_INDEX_FIELDS = (
    "row_index",
    "run_index",
    "rollout_id",
    "trajectory_filename",
    "status",
    "reward",
    "tokens",
    "duration_ms",
    "error_type",
    "resumed",
)


def test_the_emittable_key_set_matches_the_monolith_index_schema() -> None:
    # A synthetic maximal record: every optional field populated at once, which
    # no single real attempt produces, so that the row carries the whole schema.
    row = build_index_row(
        _record(0, 0, rollout_id=ROLLOUT_A, error_type="timeout"),
        trajectory_filename="trajectory.json",
        resumed=True,
    )
    assert set(row) == set(MONOLITH_INDEX_FIELDS)
    assert row == {
        "row_index": 0,
        "run_index": 0,
        "rollout_id": ROLLOUT_A,
        "trajectory_filename": "trajectory.json",
        "status": "success",
        "reward": 1.0,
        "tokens": 11,
        "duration_ms": 42.0,
        "error_type": "timeout",
        "resumed": True,
    }


def test_none_valued_fields_are_omitted_by_the_builder() -> None:
    row = build_index_row(_record(0, 0, rollout_id=ROLLOUT_A, reward=None, tokens=None))
    assert "reward" not in row
    assert "tokens" not in row
    # The platform drops a line carrying a null field silently, so a serialized
    # row must never contain one.
    assert "null" not in json.dumps(row, allow_nan=False)


def test_resumed_is_written_only_when_true() -> None:
    assert "resumed" not in build_index_row(_record(0, 0, rollout_id=ROLLOUT_A))
    assert (
        build_index_row(_record(0, 0, rollout_id=ROLLOUT_A), resumed=True)["resumed"]
        is True
    )


def test_index_lines_are_sorted_by_row_then_run() -> None:
    rows = [
        build_index_row(_record(1, 0, rollout_id=ROLLOUT_A)),
        build_index_row(_record(0, 1, rollout_id=ROLLOUT_B)),
        build_index_row(_record(0, 0, rollout_id=ROLLOUT_A)),
    ]
    keys = [
        (json.loads(line)["row_index"], json.loads(line)["run_index"])
        for line in render_index_lines(rows).decode().splitlines()
    ]
    assert keys == [(0, 0), (0, 1), (1, 0)]


# --------------------------------------------------------------------------- #
# ATIF identity (§2.3)
# --------------------------------------------------------------------------- #


def test_identity_prefers_extra_osmosis_rollout_id() -> None:
    document = _atif(ROLLOUT_A, session_id="other")
    assert atif_rollout_identity(document) == ROLLOUT_A


def test_identity_falls_back_to_session_id() -> None:
    document = {"session_id": ROLLOUT_A, "trajectory_id": "x/y"}
    assert atif_rollout_identity(document) == ROLLOUT_A


def test_identity_falls_back_to_the_trajectory_id_prefix() -> None:
    document = {"trajectory_id": f"{ROLLOUT_A}/step-1"}
    assert atif_rollout_identity(document) == ROLLOUT_A


def test_a_bare_trajectory_id_is_not_a_valid_fallback() -> None:
    # The platform converter drops this document, so the local reader must too.
    assert atif_rollout_identity({"trajectory_id": ROLLOUT_A}) is None


def test_read_valid_trajectory_accepts_a_matching_document(tmp_path: Path) -> None:
    path = _write_trajectory(tmp_path, ROLLOUT_A, _atif(ROLLOUT_A))
    assert read_valid_trajectory(path, rollout_id=ROLLOUT_A) is not None


@pytest.mark.parametrize(
    "document",
    [_atif(ROLLOUT_B), [1, 2], "text"],
    ids=["identity-mismatch", "not-an-object", "not-an-object-string"],
)
def test_read_valid_trajectory_rejects_unusable_documents(
    tmp_path: Path, document: Any
) -> None:
    path = _write_trajectory(tmp_path, ROLLOUT_A, document)
    assert read_valid_trajectory(path, rollout_id=ROLLOUT_A) is None


def test_read_valid_trajectory_tolerates_absent_and_unparseable_files(
    tmp_path: Path,
) -> None:
    assert read_valid_trajectory(tmp_path / "nope.json", rollout_id=ROLLOUT_A) is None
    broken = tmp_path / "broken.json"
    broken.write_text("{oops")
    assert read_valid_trajectory(broken, rollout_id=ROLLOUT_A) is None


# --------------------------------------------------------------------------- #
# Metrics (§2.5)
# --------------------------------------------------------------------------- #


def _row(
    row: int, run: int, status: str, reward: float | None, tokens: int = 10
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_index": row,
        "run_index": run,
        "rollout_id": ROLLOUT_A,
        "status": status,
        "duration_ms": 1.0,
        "tokens": tokens,
    }
    if reward is not None:
        payload["reward"] = reward
    return payload


def test_pass_rate_excludes_skipped_from_scored() -> None:
    summary = aggregate_metrics(
        [
            _row(0, 0, "success", 1.0),
            _row(1, 0, "success", 0.0),
            _row(2, 0, "skipped", None),
        ],
        pass_threshold=1.0,
    )
    assert summary["total_samples"] == 3
    assert summary["skipped"] == 1
    assert summary["completed_samples"] == 2
    assert summary["graded"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5


def test_passed_uses_a_greater_or_equal_threshold() -> None:
    summary = aggregate_metrics([_row(0, 0, "success", 0.7)], pass_threshold=0.7)
    assert summary["passed"] == 1


def test_a_failed_row_without_a_reward_is_not_graded() -> None:
    summary = aggregate_metrics([_row(0, 0, "failed", None)], pass_threshold=1.0)
    assert summary["failed"] == 1
    assert summary["graded"] == 0
    assert summary["passed"] == 0
    assert summary["pass_rate"] == 0
    assert "reward_stats" not in summary


def test_tokens_are_summed_across_every_row() -> None:
    summary = aggregate_metrics(
        [_row(0, 0, "success", 1.0, tokens=617), _row(1, 0, "skipped", None, tokens=3)],
        pass_threshold=1.0,
    )
    assert summary["tokens_used"] == 620


def test_reward_stats_shape_matches_the_download_projection() -> None:
    summary = aggregate_metrics(
        [_row(0, 0, "success", 0.0), _row(1, 0, "success", 1.0)], pass_threshold=1.0
    )
    assert summary["reward_stats"] == {
        "mean": 0.5,
        "median": 0.5,
        "std": 0.5,
        "min": 0.0,
        "max": 1.0,
    }


def test_single_attempt_runs_have_no_pass_at_k() -> None:
    summary = aggregate_metrics([_row(0, 0, "success", 1.0)], pass_threshold=1.0)
    assert "pass_at_k" not in summary
    assert "n_runs" not in summary


def test_pass_at_k_is_reported_for_multi_attempt_runs() -> None:
    rows = [
        _row(0, 0, "success", 1.0),
        _row(0, 1, "success", 0.0),
        _row(1, 0, "success", 0.0),
        _row(1, 1, "success", 0.0),
    ]
    summary = aggregate_metrics(rows, pass_threshold=1.0)
    assert summary["n_runs"] == 2
    assert summary["pass_at_k"] == [{"k": 1, "value": 0.25}, {"k": 2, "value": 0.5}]


@pytest.mark.parametrize(
    ("attempts", "passes", "k", "expected"),
    [
        (1, 1, 1, 1.0),
        (1, 0, 1, 0.0),
        (2, 1, 1, 0.5),
        (2, 1, 2, 1.0),
        (4, 1, 2, 0.5),
        (4, 0, 4, 0.0),
    ],
)
def test_unbiased_pass_at_k(
    attempts: int, passes: int, k: int, expected: float
) -> None:
    assert pass_at_k(attempts=attempts, passes=passes, k=k) == pytest.approx(expected)


def test_pass_at_k_rejects_k_beyond_the_attempt_count() -> None:
    with pytest.raises(ValueError, match="k must not exceed"):
        pass_at_k(attempts=1, passes=0, k=2)


def test_an_empty_index_aggregates_without_dividing_by_zero() -> None:
    summary = aggregate_metrics([], pass_threshold=1.0)
    assert summary["total_samples"] == 0
    assert summary["pass_rate"] == 0
    assert summary["tokens_used"] == 0


# --------------------------------------------------------------------------- #
# Artifact path safety (§2.6)
# --------------------------------------------------------------------------- #


def test_artifact_enumeration_skips_the_reserved_manifest_name(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "logs").mkdir(parents=True)
    (artifacts / "manifest.json").write_text("{}")
    (artifacts / "logs" / "manifest.json").write_text("{}")
    (artifacts / "out.txt").write_text("hi")
    found = {str(path) for path in safe_artifact_relative_paths(artifacts)}
    # Only the exact top-level name is reserved; a nested one is a real artifact.
    assert found == {"out.txt", "logs/manifest.json"}


def test_artifact_enumeration_skips_symlinks_without_failing_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Skipped, not fatal: every terminal result is already durable, so one stray
    # symlink must not stop the run from finalizing.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE")
    (artifacts / "leak").symlink_to(secret)
    (artifacts / "real.txt").write_text("kept")
    with caplog.at_level("WARNING"):
        found = safe_artifact_relative_paths(artifacts)
    assert [str(path) for path in found] == ["real.txt"]
    assert "symlink" in caplog.text


def test_a_symlinked_artifact_is_never_copied_into_the_projection(
    tmp_path: Path,
) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    artifacts = trials / ROLLOUT_A / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE")
    (artifacts / "leak").symlink_to(secret)

    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    projected = tmp_path / "artifacts" / "row_0_run_0"
    assert not (projected / "leak").exists()
    assert "PRIVATE" not in (tmp_path / "index.jsonl").read_text()


def test_a_symlinked_artifact_root_projects_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The per-entry checks resolve against the root itself, so a symlinked root
    # would let every file under the link's target pass as "inside".
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "id_rsa").write_text("PRIVATE")
    artifacts = tmp_path / "artifacts"
    artifacts.symlink_to(elsewhere, target_is_directory=True)
    with caplog.at_level("WARNING"):
        found = safe_artifact_relative_paths(artifacts)
    assert found == []
    assert "symlinked artifact root" in caplog.text


def test_a_missing_artifacts_dir_yields_nothing(tmp_path: Path) -> None:
    assert safe_artifact_relative_paths(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# Selection and full materialization
# --------------------------------------------------------------------------- #


def _identity() -> RunIdentity:
    return RunIdentity(
        local_run_id="c" * 32,
        run_name="my-run",
        dataset_name="multiply",
        model_name="openai/gpt-5-mini",
        rollout_name="multiply-local-openai",
        started_at="2026-08-14T00:00:00Z",
    )


def test_select_attempts_binds_a_valid_trajectory(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    [attempt] = select_attempts(latest, trials_dir=trials)
    assert attempt.trajectory_filename == "trajectory.json"
    assert attempt.resumed is False


def test_select_attempts_omits_a_never_written_trajectory(tmp_path: Path) -> None:
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    [attempt] = select_attempts(latest, trials_dir=tmp_path / "rollout_trials")
    # A missing trajectory never invalidates the terminal result (§11.3).
    assert attempt.trajectory_filename is None
    assert attempt.record.status == "success"


def test_select_attempts_marks_replayed_keys_as_resumed(tmp_path: Path) -> None:
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    [attempt] = select_attempts(
        latest, trials_dir=tmp_path / "rollout_trials", resumed_keys=[(0, 0)]
    )
    assert attempt.resumed is True


def test_a_late_trajectory_is_picked_up_on_the_next_refresh(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    assert select_attempts(latest, trials_dir=trials)[0].trajectory_filename is None
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    assert (
        select_attempts(latest, trials_dir=trials)[0].trajectory_filename
        == "trajectory.json"
    )


def test_refresh_writes_the_full_download_layout(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    (trials / ROLLOUT_A / "artifacts" / "logs").mkdir(parents=True)
    (trials / ROLLOUT_A / "artifacts" / "logs" / "run.log").write_text("hello")

    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    attempts = select_attempts(latest, trials_dir=trials)
    Materializer(tmp_path).refresh(
        attempts,
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1000,
        total_runs=1,
    )

    index = (tmp_path / "index.jsonl").read_text()
    assert json.loads(index)["trajectory_filename"] == "trajectory.json"
    # summary.jsonl is index.jsonl verbatim.
    assert (tmp_path / "summary.jsonl").read_text() == index
    assert json.loads((tmp_path / "progress.json").read_text()) == {
        "total_runs": 1,
        "sampled_rows": 1,
        "total_dataset_rows": 1000,
    }
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["eval_run"]["name"] == "my-run"
    assert metrics["summary"]["pass_rate"] == 1
    assert (tmp_path / "trajectories" / "row_0_run_0.json").is_file()
    assert (
        tmp_path / "artifacts" / "row_0_run_0" / "logs" / "run.log"
    ).read_text() == "hello"


def test_metrics_json_uses_two_space_json_with_a_trailing_newline(
    tmp_path: Path,
) -> None:
    Materializer(tmp_path).refresh(
        select_attempts({}, trials_dir=tmp_path / "rollout_trials"),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=0,
        total_dataset_rows=0,
        total_runs=0,
    )
    text = (tmp_path / "metrics.json").read_text()
    assert text.endswith("}\n")
    assert '\n  "eval_run"' in text


def test_projections_are_independent_copies(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    canonical = _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}
    Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    projection = tmp_path / "trajectories" / "row_0_run_0.json"
    projection.write_text('{"edited": true}')
    # Editing the projection must never mutate the canonical upload source.
    assert json.loads(canonical.read_text())["session_id"] == ROLLOUT_A


def test_a_retried_work_item_projects_only_the_selected_attempt(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    _write_trajectory(trials, ROLLOUT_B, _atif(ROLLOUT_B))
    # The journal's latest record for (0, 0) points at the second attempt.
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_B)}
    rows = Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    assert [row["rollout_id"] for row in rows] == [ROLLOUT_B]
    # The superseded attempt stays on disk as diagnostic evidence.
    assert (trials / ROLLOUT_A / "trajectory.json").is_file()


def test_a_cancelled_attempt_is_never_projected_as_success(tmp_path: Path) -> None:
    # The Harbor footgun: a cancelled attempt writes no terminal record, so it
    # simply is not in the index at all.
    rows = Materializer(tmp_path).refresh(
        select_attempts({}, trials_dir=tmp_path / "rollout_trials"),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    assert rows == []
    assert (tmp_path / "index.jsonl").read_text() == ""


# --------------------------------------------------------------------------- #
# Review fixes: pass-rate denominator, projection pruning, token fallback
# --------------------------------------------------------------------------- #


def test_reward_less_failures_stay_in_the_pass_rate_denominator() -> None:
    # Excluding them would report pass_rate 1.0 for a run where half the rows
    # failed -- the metric would hide exactly what the user needs to see.
    summary = aggregate_metrics(
        [_row(0, 0, "success", 1.0), _row(1, 0, "failed", None)], pass_threshold=1.0
    )
    assert summary["graded"] == 1
    assert summary["completed_samples"] == 2
    assert summary["pass_rate"] == 0.5


def test_skipped_rows_are_the_only_thing_excluded_from_scored() -> None:
    summary = aggregate_metrics(
        [
            _row(0, 0, "success", 1.0),
            _row(1, 0, "failed", None),
            _row(2, 0, "skipped", None),
        ],
        pass_threshold=1.0,
    )
    assert summary["completed_samples"] == 2
    assert summary["pass_rate"] == 0.5


def test_pass_at_k_counts_a_reward_less_failure_as_a_non_pass() -> None:
    rows = [
        _row(0, 0, "success", 1.0),
        _row(0, 1, "failed", None),
        _row(1, 0, "failed", None),
        _row(1, 1, "failed", None),
    ]
    summary = aggregate_metrics(rows, pass_threshold=1.0)
    assert summary["pass_at_k"] == [{"k": 1, "value": 0.25}, {"k": 2, "value": 0.5}]


def test_a_retry_without_a_trajectory_clears_the_superseded_projection(
    tmp_path: Path,
) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    (trials / ROLLOUT_A / "artifacts").mkdir(parents=True)
    (trials / ROLLOUT_A / "artifacts" / "out.txt").write_text("attempt A")
    materializer = Materializer(tmp_path)
    kwargs: dict[str, Any] = {
        "identity": _identity(),
        "pass_threshold": 1.0,
        "sampled_rows": 1,
        "total_dataset_rows": 1,
        "total_runs": 1,
    }
    materializer.refresh(
        select_attempts(
            {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A)}, trials_dir=trials
        ),
        **kwargs,
    )
    assert (tmp_path / "trajectories" / "row_0_run_0.json").is_file()
    assert (tmp_path / "artifacts" / "row_0_run_0" / "out.txt").is_file()

    # Attempt B times out: no trajectory, no artifacts. The stem is per work
    # item, so A's files must not survive attributed to B's result.
    retried = _record(0, 0, rollout_id=ROLLOUT_B, status="failed", reward=None)
    rows = materializer.refresh(
        select_attempts({(0, 0): retried}, trials_dir=trials), **kwargs
    )
    assert "trajectory_filename" not in rows[0]
    assert not (tmp_path / "trajectories" / "row_0_run_0.json").exists()
    assert not (tmp_path / "artifacts" / "row_0_run_0").exists()


def test_refresh_projects_only_the_requested_work_items(tmp_path: Path) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    _write_trajectory(trials, ROLLOUT_B, _atif(ROLLOUT_B))
    latest = {
        (0, 0): _record(0, 0, rollout_id=ROLLOUT_A),
        (1, 0): _record(1, 0, rollout_id=ROLLOUT_B),
    }
    Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=2,
        total_dataset_rows=2,
        total_runs=2,
        project_keys=[(1, 0)],
    )
    # Both rows reach index.jsonl; only the requested one is copied.
    assert len((tmp_path / "index.jsonl").read_text().splitlines()) == 2
    assert not (tmp_path / "trajectories" / "row_0_run_0.json").exists()
    assert (tmp_path / "trajectories" / "row_1_run_0.json").is_file()


def test_tokens_come_from_the_trajectorys_final_metrics(
    tmp_path: Path,
) -> None:
    trials = tmp_path / "rollout_trials"
    document = _atif(ROLLOUT_A)
    document["final_metrics"] = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 23,
    }
    _write_trajectory(trials, ROLLOUT_A, document)
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A, tokens=None)}
    rows = Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    assert rows[0]["tokens"] == 123


def test_a_trajectory_without_final_metrics_leaves_tokens_absent(
    tmp_path: Path,
) -> None:
    trials = tmp_path / "rollout_trials"
    _write_trajectory(trials, ROLLOUT_A, _atif(ROLLOUT_A))
    latest = {(0, 0): _record(0, 0, rollout_id=ROLLOUT_A, tokens=None)}
    rows = Materializer(tmp_path).refresh(
        select_attempts(latest, trials_dir=trials),
        identity=_identity(),
        pass_threshold=1.0,
        sampled_rows=1,
        total_dataset_rows=1,
        total_runs=1,
    )
    assert "tokens" not in rows[0]
