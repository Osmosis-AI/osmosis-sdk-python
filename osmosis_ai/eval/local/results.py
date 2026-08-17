"""Result materialization: run snapshots and user projections.

Every file here is a **projection** rebuilt from the terminal journal and the
canonical ``rollout_trials/`` tree; none of it decides whether work reruns
(design ``local-eval-run-plan.md`` §11).
"""

from __future__ import annotations

import json
import logging
import shutil
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.eval.local.state import (
    TerminalRecord,
    WorkKey,
    atomic_write_bytes,
    atomic_write_json,
    drop_none_values,
)

CANONICAL_TRAJECTORY_FILENAME = "trajectory.json"
INDEX_FILENAME = "index.jsonl"
PROGRESS_FILENAME = "progress.json"
SUMMARY_FILENAME = "summary.jsonl"
METRICS_FILENAME = "metrics.json"
TRIALS_DIRNAME = "rollout_trials"
TRAJECTORIES_DIRNAME = "trajectories"
ARTIFACTS_DIRNAME = "artifacts"

#: Reserved on both sides of the download contract (§2.6).
_RESERVED_ARTIFACT_MANIFEST = "manifest.json"

logger: logging.Logger = logging.getLogger(__name__)


class ArtifactProjectionError(RuntimeError):
    """An artifact path is unsafe to copy into the user projection."""


# --------------------------------------------------------------------------- #
# index.jsonl
# --------------------------------------------------------------------------- #


def build_index_row(
    record: TerminalRecord,
    *,
    trajectory_filename: str | None = None,
    resumed: bool = False,
    tokens: int | None = None,
) -> dict[str, Any]:
    """Project one terminal record into an ``index.jsonl`` row.

    ``resumed`` marks a result carried forward from an earlier invocation of the
    same named run, matching the cloud controller's carry-forward flag.

    The key set emitted here is intended to stay in parity with the monolith eval
    controller's index schema, because the platform drops a malformed
    ``index.jsonl`` line *silently* (§2.2). Validity is held by construction
    rather than by a write-time check: ``TerminalRecord`` is frozen with an
    integer row/run index, a ``Literal`` status, a ``uuid4().hex`` rollout id and
    a non-optional ``duration_ms``; ``trajectory_filename`` is only ever the
    canonical name; and ``None`` values are dropped instead of written as
    ``null``.
    """
    return drop_none_values(
        {
            "row_index": record.row_index,
            "run_index": record.run_index,
            "rollout_id": record.rollout_id,
            "trajectory_filename": trajectory_filename,
            "status": record.status,
            "reward": record.reward,
            "tokens": record.tokens if tokens is None else tokens,
            "duration_ms": record.duration_ms,
            "error_type": record.error_type,
            "resumed": True if resumed else None,
        }
    )


def render_index_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize index rows, sorted by ``(row_index, run_index)``."""
    ordered = sorted(rows, key=lambda row: (row["row_index"], row["run_index"]))
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in ordered
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# ATIF identity
# --------------------------------------------------------------------------- #


def atif_rollout_identity(document: Mapping[str, Any]) -> str | None:
    """Resolve the rollout id the platform converter will read from an ATIF doc.

    Order per §2.3: ``extra.osmosis.rollout_id``, then top-level ``session_id``,
    then the prefix before the first ``/`` in ``trajectory_id``. A bare
    ``trajectory_id`` with no ``/`` is **not** a valid fallback -- the platform
    converter drops that document, so accepting it locally would let a run look
    healthy and upload blind.
    """
    extra = document.get("extra")
    if isinstance(extra, Mapping):
        osmosis = extra.get("osmosis")
        if isinstance(osmosis, Mapping):
            candidate = osmosis.get("rollout_id")
            if isinstance(candidate, str) and candidate:
                return candidate
    session_id = document.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    trajectory_id = document.get("trajectory_id")
    if isinstance(trajectory_id, str) and "/" in trajectory_id:
        prefix = trajectory_id.split("/", 1)[0]
        if prefix:
            return prefix
    return None


def read_valid_trajectory(path: Path, *, rollout_id: str) -> dict[str, Any] | None:
    """Return the ATIF document at *path* when it parses and its identity matches.

    ``None`` covers every reason the platform would ignore the file: absent,
    unparseable, not a single JSON object, or an identity that disagrees with the
    directory name. The terminal result stays valid either way (§11.3).
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if atif_rollout_identity(payload) != rollout_id:
        return None
    return payload


# --------------------------------------------------------------------------- #
# Metrics (§2.5)
# --------------------------------------------------------------------------- #


def pass_at_k(*, attempts: int, passes: int, k: int) -> float:
    """Unbiased pass@k for one work row (Chen et al.)."""
    if k > attempts:
        raise ValueError("k must not exceed the attempt count")
    if attempts - passes < k:
        return 1.0
    value = 1.0
    for index in range(k):
        value *= (attempts - passes - index) / (attempts - index)
    return 1.0 - value


def _powers_of_two_up_to(limit: int) -> list[int]:
    values: list[int] = []
    k = 1
    while k <= limit:
        values.append(k)
        k *= 2
    return values


def aggregate_metrics(
    rows: Iterable[Mapping[str, Any]], *, pass_threshold: float
) -> dict[str, Any]:
    """Replicate the platform worker's ``_aggregate_from_index`` summary (§2.5).

    ``scored`` is every non-skipped sample; ``passed`` is
    ``reward >= pass_threshold``. Platform finalize always recomputes this, so a
    local number is a display convenience -- but it must agree, or a user sees
    two truths.
    """
    materialized = list(rows)
    passes_by_row: dict[int, int] = {}
    attempts_by_row: dict[int, int] = {}
    rewards: list[float] = []
    passed = graded = skipped = failed = 0
    tokens_used = 0
    max_run_index = -1

    for row in materialized:
        status = row.get("status")
        run_index = row.get("run_index")
        if isinstance(run_index, int):
            max_run_index = max(max_run_index, run_index)
        tokens = row.get("tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            tokens_used += tokens
        if status == "skipped":
            skipped += 1
            continue
        if status == "failed":
            failed += 1
        row_index = row.get("row_index")
        reward = row.get("reward")
        # A failed attempt usually carries no reward. It is still a non-pass in a
        # denominator that excludes only skipped rows: dropping it would let a run
        # where most rows failed report the pass rate of the rows that worked.
        is_pass = False
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            graded += 1
            rewards.append(float(reward))
            is_pass = float(reward) >= pass_threshold
            if is_pass:
                passed += 1
        if isinstance(row_index, int):
            attempts_by_row[row_index] = attempts_by_row.get(row_index, 0) + 1
            passes_by_row[row_index] = passes_by_row.get(row_index, 0) + int(is_pass)

    total_samples = len(materialized)
    # ``scored`` excludes skipped rows only (§2.5), which is exactly
    # ``completed_samples``.
    scored = total_samples - skipped
    summary: dict[str, Any] = {
        "total_samples": total_samples,
        "completed_samples": scored,
        "graded": graded,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": (passed / scored) if scored else 0,
        "pass_threshold": pass_threshold,
        "tokens_used": tokens_used,
    }
    if rewards:
        summary["reward_stats"] = {
            "mean": statistics.fmean(rewards),
            "median": statistics.median(rewards),
            "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "min": min(rewards),
            "max": max(rewards),
        }

    n_runs = max_run_index + 1
    if n_runs >= 2:
        summary["n_runs"] = n_runs
        # pass@k needs at least two points to be worth plotting, so it appears
        # only for multi-attempt runs.
        points: list[dict[str, Any]] = []
        for k in _powers_of_two_up_to(n_runs):
            eligible = [
                pass_at_k(
                    attempts=attempts_by_row[row_index],
                    passes=passes_by_row[row_index],
                    k=k,
                )
                for row_index in sorted(attempts_by_row)
                if attempts_by_row[row_index] >= k
            ]
            if eligible:
                points.append({"k": k, "value": statistics.fmean(eligible)})
        if len(points) >= 2:
            summary["pass_at_k"] = points
    return summary


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #


def projection_stem(row_index: int, run_index: int) -> str:
    return f"row_{row_index}_run_{run_index}"


def safe_artifact_relative_paths(artifacts_dir: Path) -> list[Path]:
    """Enumerate copyable artifact paths, refusing anything unsafe.

    ``rollout_trials/<id>/artifacts/**`` is the one tree the platform renders
    wholesale, so a symlink or traversal here would exfiltrate host files into a
    user-visible projection and, later, into an upload.
    """
    if not artifacts_dir.is_dir():
        return []
    resolved_root = artifacts_dir.resolve()
    selected: list[Path] = []
    for candidate in sorted(artifacts_dir.rglob("*")):
        relative = candidate.relative_to(artifacts_dir)
        # An unsafe entry is skipped with a warning, never projected, and never
        # fatal: one stray symlink must not stop a run whose every terminal
        # result is already durable.
        if candidate.is_symlink():
            logger.warning("skipping artifact symlink %s", candidate)
            continue
        if not candidate.is_file():
            continue
        if any(part in ("", ".", "..") for part in relative.parts):
            logger.warning("skipping unsafe artifact path %s", relative)
            continue
        if len(relative.parts) == 1 and relative.name == _RESERVED_ARTIFACT_MANIFEST:
            # Reserved server-side index name; excluded on both sides (§2.6).
            continue
        if not candidate.resolve().is_relative_to(resolved_root):
            logger.warning(
                "skipping artifact path %s outside the artifact root", relative
            )
            continue
        selected.append(relative)
    return selected


def copy_projection_file(source: Path, destination: Path) -> None:
    """Copy bytes into the user projection, never linking to the upload source.

    Editing ``trajectories/row_*`` must not mutate
    ``rollout_trials/<id>/trajectory.json``, so this is always an independent
    copy written through the atomic helper.
    """
    atomic_write_bytes(destination, source.read_bytes())


# --------------------------------------------------------------------------- #
# Snapshot writer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunIdentity:
    """The ``eval_run`` block of ``metrics.json``: layer-2 display only."""

    local_run_id: str
    run_name: str
    dataset_name: str
    model_name: str
    rollout_name: str
    started_at: str
    status: str = "running"
    completed_at: str | None = None
    duration_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return drop_none_values(
            {
                "id": self.local_run_id,
                "status": self.status,
                "name": self.run_name,
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "rollout_name": self.rollout_name,
                "duration_ms": self.duration_ms,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            }
        )


@dataclass(frozen=True)
class SelectedAttempt:
    """One work item's selected terminal attempt, with its trajectory verdict."""

    record: TerminalRecord
    resumed: bool
    trajectory_filename: str | None
    tokens: int | None = None

    @property
    def key(self) -> WorkKey:
        return self.record.key


def atif_total_tokens(document: Mapping[str, Any]) -> int | None:
    """Token total from an ATIF document's ``final_metrics`` (§7.2).

    The trajectory is where a local run reads its tokens: totals the platform
    already trusts, out of a file that is written anyway.
    """
    metrics = document.get("final_metrics")
    if not isinstance(metrics, Mapping):
        return None
    total = 0
    seen = False
    for key in ("total_prompt_tokens", "total_completion_tokens"):
        value = metrics.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
            seen = True
    return total if seen else None


def select_attempts(
    latest: Mapping[WorkKey, TerminalRecord],
    *,
    trials_dir: Path,
    resumed_keys: Iterable[WorkKey] = (),
) -> list[SelectedAttempt]:
    """Bind each work item's terminal record to a verified trajectory, if any.

    Re-checked against the filesystem on every refresh, so a trajectory written
    just before a crash is picked up on restart with no extra state event
    (§11.3).
    """
    resumed = set(resumed_keys)
    selected: list[SelectedAttempt] = []
    for key in sorted(latest):
        record = latest[key]
        trajectory_path = trials_dir / record.rollout_id / CANONICAL_TRAJECTORY_FILENAME
        document = read_valid_trajectory(trajectory_path, rollout_id=record.rollout_id)
        tokens = record.tokens
        if tokens is None and document is not None:
            tokens = atif_total_tokens(document)
        selected.append(
            SelectedAttempt(
                record=record,
                resumed=key in resumed,
                trajectory_filename=(
                    CANONICAL_TRAJECTORY_FILENAME if document is not None else None
                ),
                tokens=tokens,
            )
        )
    return selected


class Materializer:
    """Writes every projection for one run directory.

    Snapshots are refreshed while the run is live, so partial output must always
    be readable: every file goes through the atomic write helper.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    @property
    def trials_dir(self) -> Path:
        return self._run_dir / TRIALS_DIRNAME

    def refresh(
        self,
        attempts: Sequence[SelectedAttempt],
        *,
        identity: RunIdentity,
        pass_threshold: float,
        sampled_rows: int,
        total_dataset_rows: int,
        total_runs: int,
        project_keys: Iterable[WorkKey] | None = None,
    ) -> list[dict[str, Any]]:
        """Rewrite index, progress, summary, metrics, and both projections.

        Snapshots are cheap to rewrite whole; file projections are not. Pass
        *project_keys* to copy only the work items whose selected attempt
        changed -- copying every attempt on every refresh is quadratic in the
        row count and re-fsyncs the entire trajectory set each time.
        """
        rows = [
            build_index_row(
                attempt.record,
                trajectory_filename=attempt.trajectory_filename,
                resumed=attempt.resumed,
                tokens=attempt.tokens,
            )
            for attempt in attempts
        ]
        payload = render_index_lines(rows)
        atomic_write_bytes(self._run_dir / INDEX_FILENAME, payload)
        # summary.jsonl is index.jsonl verbatim, not a second schema (§2.4).
        atomic_write_bytes(self._run_dir / SUMMARY_FILENAME, payload)
        atomic_write_json(
            self._run_dir / PROGRESS_FILENAME,
            {
                "total_runs": total_runs,
                "sampled_rows": sampled_rows,
                "total_dataset_rows": total_dataset_rows,
            },
        )
        atomic_write_json(
            self._run_dir / METRICS_FILENAME,
            {
                "eval_run": identity.to_payload(),
                "summary": aggregate_metrics(rows, pass_threshold=pass_threshold),
            },
        )
        wanted = None if project_keys is None else set(project_keys)
        for attempt in attempts:
            if wanted is None or attempt.key in wanted:
                self.project_attempt(attempt)
        return rows

    def project_attempt(self, attempt: SelectedAttempt) -> None:
        """Copy the selected attempt's trajectory and artifacts to top level.

        The previous attempt's projection is cleared first: the stem is per work
        item, so a retry whose new attempt produced no trajectory would otherwise
        leave the superseded attempt's files on disk, attributed to a result that
        did not produce them (§9.2, §11.1).
        """
        record = attempt.record
        stem = projection_stem(record.row_index, record.run_index)
        rollout_dir = self.trials_dir / record.rollout_id
        trajectory_dest = self._run_dir / TRAJECTORIES_DIRNAME / f"{stem}.json"
        artifacts_dest = self._run_dir / ARTIFACTS_DIRNAME / stem
        trajectory_dest.unlink(missing_ok=True)
        shutil.rmtree(artifacts_dest, ignore_errors=True)
        if attempt.trajectory_filename is not None:
            copy_projection_file(
                rollout_dir / attempt.trajectory_filename, trajectory_dest
            )
        artifacts_dir = rollout_dir / ARTIFACTS_DIRNAME
        for relative in safe_artifact_relative_paths(artifacts_dir):
            copy_projection_file(artifacts_dir / relative, artifacts_dest / relative)
