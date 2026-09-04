"""Validate and describe a completed local evaluation for platform import.

This module is deliberately local and side-effect free: it reads the run
directory, validates the immutable upload contract, and hashes selected files.
Network I/O and CLI rendering live in :mod:`osmosis_ai.platform.cli.eval_upload`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, TypeGuard

from osmosis_ai.eval.local.results import atif_rollout_identity
from osmosis_ai.rollout.trajectory.atif import Trajectory

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_TRAJECTORY_NAME = re.compile(r"^trajectory[A-Za-z0-9._-]*\.json$")
_TERMINAL_STATUSES = frozenset({"success", "failed", "skipped"})
_PROVENANCE_KEYS = (
    "sdk_version",
    "git_head",
    "git_branch",
    "git_dirty",
    "config_branch",
    "config_commit_sha",
)
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_UPLOAD_PATH_LENGTH = 1024


class LocalEvalUploadError(RuntimeError):
    """A local run directory cannot be safely imported."""


@dataclass(frozen=True)
class EvalUploadFile:
    """One selected local file and the immutable identity sent to the server."""

    path: str
    source: Path
    size: int
    sha256: str
    device: int
    inode: int
    modified_ns: int
    changed_ns: int

    def to_request(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @contextmanager
    def open_verified(self) -> Iterator[IO[bytes]]:
        """Open the exact regular file identity captured by the upload plan."""
        with _open_regular_file(self.source, where=self.path) as (handle, opened):
            if _stat_signature(opened) != (
                self.device,
                self.inode,
                self.size,
                self.modified_ns,
                self.changed_ns,
            ):
                raise LocalEvalUploadError(
                    f"{self.path} changed after the upload plan was built; retry"
                )
            yield handle


@dataclass(frozen=True)
class EvalUploadPlan:
    """Validated metadata and files for one completed local evaluation."""

    run_dir: Path
    local_run_id: str
    manifest_digest: str
    run: dict[str, Any]
    schema_versions: dict[str, Any]
    provenance: dict[str, Any]
    files: tuple[EvalUploadFile, ...]

    def file_requests(self) -> list[dict[str, Any]]:
        return [file.to_request() for file in self.files]


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _object(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LocalEvalUploadError(f"{where} must be a JSON object")
    return value


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalEvalUploadError(f"{where} must be a non-empty string")
    return value


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _open_regular_file(
    path: Path, *, where: str
) -> Iterator[tuple[IO[bytes], os.stat_result]]:
    """Open one stable regular file without trusting a path after validation."""
    try:
        named_before = path.lstat()
    except OSError as exc:
        raise LocalEvalUploadError(f"{where} is missing or unreadable: {exc}") from exc
    if not stat.S_ISREG(named_before.st_mode):
        raise LocalEvalUploadError(f"{where} must be a regular, non-symlink file")

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise LocalEvalUploadError(f"{where} is unreadable: {exc}") from exc
    try:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or _stat_signature(
            named_before
        ) != _stat_signature(opened):
            raise LocalEvalUploadError(f"{where} changed while being opened; retry")
        yield handle, opened
        opened_after = os.fstat(handle.fileno())
        try:
            named_after = path.lstat()
        except OSError as exc:
            raise LocalEvalUploadError(
                f"{where} changed while being read; retry"
            ) from exc
        if (
            not stat.S_ISREG(named_after.st_mode)
            or _stat_signature(opened) != _stat_signature(opened_after)
            or _stat_signature(opened_after) != _stat_signature(named_after)
        ):
            raise LocalEvalUploadError(f"{where} changed while being read; retry")
    finally:
        handle.close()


def _read_bytes(path: Path, *, where: str) -> bytes:
    try:
        with _open_regular_file(path, where=where) as (handle, _opened):
            return handle.read()
    except OSError as exc:
        raise LocalEvalUploadError(f"{where} is unreadable: {exc}") from exc


def _json_object(path: Path, *, where: str) -> dict[str, Any]:
    raw = _read_bytes(path, where=where)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalEvalUploadError(f"{where} is not valid JSON: {exc}") from exc
    return _object(value, where=where)


def _hash_file(path: Path, *, relative: str) -> EvalUploadFile:
    digest = hashlib.sha256()
    try:
        with _open_regular_file(path, where=relative) as (handle, opened):
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise LocalEvalUploadError(f"{relative} is unreadable: {exc}") from exc
    return EvalUploadFile(
        path=relative,
        source=path,
        size=opened.st_size,
        sha256=digest.hexdigest(),
        device=opened.st_dev,
        inode=opened.st_ino,
        modified_ns=opened.st_mtime_ns,
        changed_ns=opened.st_ctime_ns,
    )


def _validate_upload_path(path: str, *, where: str) -> None:
    if len(path) > _MAX_UPLOAD_PATH_LENGTH:
        raise LocalEvalUploadError(
            f"{where} upload path exceeds {_MAX_UPLOAD_PATH_LENGTH} characters"
        )
    if "\\" in path or any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise LocalEvalUploadError(f"{where} contains a backslash or control character")
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise LocalEvalUploadError(f"{where} contains an unsafe path segment")
    if PurePosixPath(path).as_posix() != path:
        raise LocalEvalUploadError(f"{where} is not a canonical POSIX path")


def _read_index(path: Path) -> list[dict[str, Any]]:
    raw = _read_bytes(path, where="index.jsonl")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalEvalUploadError(f"index.jsonl is not UTF-8: {exc}") from exc
    rows: list[dict[str, Any]] = []
    work_keys: set[tuple[int, int]] = set()
    rollout_ids: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise LocalEvalUploadError(f"index.jsonl:{lineno} is blank")
        try:
            row = _object(json.loads(line), where=f"index.jsonl:{lineno}")
        except json.JSONDecodeError as exc:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} is not valid JSON: {exc}"
            ) from exc
        row_index = row.get("row_index")
        run_index = row.get("run_index")
        if not _is_int(row_index) or row_index < 0:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} row_index must be a nonnegative integer"
            )
        if not _is_int(run_index) or run_index < 0:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} run_index must be a nonnegative integer"
            )
        row_index_value = int(row_index)
        run_index_value = int(run_index)
        key = (row_index_value, run_index_value)
        if key in work_keys:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} duplicates row/run pair {key}"
            )
        work_keys.add(key)
        rollout_id = row.get("rollout_id")
        if not isinstance(rollout_id, str) or _HEX_32.fullmatch(rollout_id) is None:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} rollout_id must be 32 lowercase hex characters"
            )
        if rollout_id in rollout_ids:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} duplicates rollout_id {rollout_id}"
            )
        rollout_ids.add(rollout_id)
        if row.get("status") not in _TERMINAL_STATUSES:
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} status must be success, failed, or skipped"
            )
        trajectory_name = row.get("trajectory_filename")
        if trajectory_name is not None and (
            not isinstance(trajectory_name, str)
            or _TRAJECTORY_NAME.fullmatch(trajectory_name) is None
            or Path(trajectory_name).name != trajectory_name
        ):
            raise LocalEvalUploadError(
                f"index.jsonl:{lineno} trajectory_filename must be a safe "
                "trajectory*.json filename"
            )
        rows.append(row)
    return rows


def _assert_index_matches_progress(
    rows: list[dict[str, Any]],
    *,
    sampled_rows: int,
    total_dataset_rows: int,
    n: int,
    total_runs: int,
) -> None:
    if total_runs != sampled_rows * n:
        raise LocalEvalUploadError(
            "progress.json total_runs must equal sampled_rows multiplied by "
            "manifest.json inputs.n"
        )
    by_row: dict[int, set[int]] = {}
    for row in rows:
        by_row.setdefault(int(row["row_index"]), set()).add(int(row["run_index"]))
    if len(by_row) != sampled_rows:
        raise LocalEvalUploadError(
            "index.jsonl selected row count does not match progress.json sampled_rows"
        )
    if sampled_rows > total_dataset_rows:
        raise LocalEvalUploadError(
            "progress.json sampled_rows exceeds total_dataset_rows"
        )
    if set(by_row) != set(range(sampled_rows)):
        raise LocalEvalUploadError(
            "index.jsonl row_index values must equal the selected row range"
        )
    expected_runs = set(range(n))
    for row_index, run_indices in by_row.items():
        if run_indices != expected_runs:
            raise LocalEvalUploadError(
                f"index.jsonl row {row_index} does not contain every configured run_index"
            )


def _trajectory_path(run_dir: Path, row: dict[str, Any]) -> tuple[str, Path] | None:
    trajectory_name = row.get("trajectory_filename")
    if trajectory_name is None:
        return None
    rollout_id = row["rollout_id"]
    trial_dir = run_dir / "rollout_trials" / rollout_id
    if trial_dir.is_symlink() or not trial_dir.is_dir():
        raise LocalEvalUploadError(
            f"rollout_trials/{rollout_id} must be a regular, non-symlink directory"
        )
    relative = f"rollout_trials/{rollout_id}/{trajectory_name}"
    path = trial_dir / trajectory_name
    payload = _json_object(path, where=relative)
    try:
        Trajectory.model_validate(payload, strict=True)
    except ValueError as exc:
        raise LocalEvalUploadError(f"{relative} is not valid ATIF") from exc
    if atif_rollout_identity(payload) != rollout_id:
        raise LocalEvalUploadError(
            f"{relative} does not identify rollout_id {rollout_id}"
        )
    extra = payload.get("extra")
    osmosis = extra.get("osmosis") if isinstance(extra, dict) else None
    request_fields = (
        osmosis.get("request_extra_fields") if isinstance(osmosis, dict) else None
    )
    if (
        not isinstance(request_fields, dict)
        or not _is_int(request_fields.get("row_index"))
        or not _is_int(request_fields.get("run_index"))
        or request_fields["row_index"] != row["row_index"]
        or request_fields["run_index"] != row["run_index"]
    ):
        raise LocalEvalUploadError(
            f"{relative} row/run identity does not match index.jsonl"
        )
    return relative, path


def _artifact_paths(run_dir: Path, rollout_id: str) -> list[tuple[str, Path]]:
    trial_dir = run_dir / "rollout_trials" / rollout_id
    if trial_dir.is_symlink():
        raise LocalEvalUploadError(
            f"rollout_trials/{rollout_id} must be a regular, non-symlink directory"
        )
    if not trial_dir.exists():
        return []
    if not trial_dir.is_dir():
        raise LocalEvalUploadError(
            f"rollout_trials/{rollout_id} must be a regular, non-symlink directory"
        )
    root = trial_dir / "artifacts"
    if root.is_symlink():
        raise LocalEvalUploadError(
            f"rollout_trials/{rollout_id}/artifacts must be a regular, "
            "non-symlink directory"
        )
    if not root.exists():
        return []
    if not root.is_dir():
        raise LocalEvalUploadError(
            f"rollout_trials/{rollout_id}/artifacts must be a regular, "
            "non-symlink directory"
        )
    selected: list[tuple[str, Path]] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if candidate.is_symlink():
            raise LocalEvalUploadError(
                f"rollout_trials/{rollout_id}/artifacts/{relative.as_posix()} "
                "must not be a symlink"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            continue
        if any(part in ("", ".", "..") for part in relative.parts):
            raise LocalEvalUploadError(
                f"rollout_trials/{rollout_id}/artifacts contains an unsafe path"
            )
        if len(relative.parts) == 1 and relative.name == "manifest.json":
            continue
        upload_path = f"rollout_trials/{rollout_id}/artifacts/{relative.as_posix()}"
        _validate_upload_path(upload_path, where=f"artifact {relative.as_posix()!r}")
        selected.append((upload_path, candidate))
    return selected


def build_eval_upload_plan(run_dir: Path) -> EvalUploadPlan:
    """Validate *run_dir* and return its exact, sorted platform import plan."""
    run_dir = run_dir.expanduser()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise LocalEvalUploadError(
            f"{run_dir} must be a regular, non-symlink run directory"
        )
    run_dir = run_dir.resolve()
    trials_root = run_dir / "rollout_trials"
    if trials_root.is_symlink():
        raise LocalEvalUploadError(
            "rollout_trials must be a regular, non-symlink directory"
        )

    manifest_path = run_dir / "manifest.json"
    manifest_raw = _read_bytes(manifest_path, where="manifest.json")
    try:
        manifest = _object(json.loads(manifest_raw), where="manifest.json")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalEvalUploadError(f"manifest.json is not valid JSON: {exc}") from exc
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    if manifest.get("schema_version") != 1:
        raise LocalEvalUploadError("manifest.json schema_version must be 1")
    local_run_id = manifest.get("local_run_id")
    if not isinstance(local_run_id, str) or _HEX_32.fullmatch(local_run_id) is None:
        raise LocalEvalUploadError(
            "manifest.json local_run_id must be 32 lowercase hex characters"
        )
    run_name = _string(manifest.get("run_name"), where="manifest.json run_name")
    if run_name != run_dir.name:
        raise LocalEvalUploadError(
            f"manifest.json run_name {run_name!r} does not match directory "
            f"name {run_dir.name!r}"
        )
    inputs = _object(manifest.get("inputs"), where="manifest.json inputs")
    rollout = _object(inputs.get("rollout"), where="manifest.json inputs.rollout")
    dataset = _object(inputs.get("dataset"), where="manifest.json inputs.dataset")
    model_path = _string(
        inputs.get("model_path"), where="manifest.json inputs.model_path"
    )
    rollout_name = _string(
        rollout.get("name"), where="manifest.json inputs.rollout.name"
    )
    entrypoint = _string(
        rollout.get("entrypoint"), where="manifest.json inputs.rollout.entrypoint"
    )
    rollout_source_digest = _string(
        rollout.get("source_digest"),
        where="manifest.json inputs.rollout.source_digest",
    )
    dataset_sha256 = _string(
        dataset.get("sha256"), where="manifest.json inputs.dataset.sha256"
    )
    selected_source_rows = _string(
        dataset.get("selected_source_rows"),
        where="manifest.json inputs.dataset.selected_source_rows",
    )
    n = inputs.get("n")
    if not _is_int(n) or n <= 0:
        raise LocalEvalUploadError("manifest.json inputs.n must be a positive integer")
    n_value = int(n)
    schema_versions = _object(
        inputs.get("versions"), where="manifest.json inputs.versions"
    )

    progress = _json_object(run_dir / "progress.json", where="progress.json")
    total_runs = progress.get("total_runs")
    if not _is_int(total_runs) or total_runs < 0:
        raise LocalEvalUploadError(
            "progress.json total_runs must be a nonnegative integer"
        )
    total_runs_value = int(total_runs)
    sampled_rows = progress.get("sampled_rows")
    if not _is_int(sampled_rows) or sampled_rows < 0:
        raise LocalEvalUploadError(
            "progress.json sampled_rows must be a nonnegative integer"
        )
    total_dataset_rows = progress.get("total_dataset_rows")
    if not _is_int(total_dataset_rows) or total_dataset_rows < 0:
        raise LocalEvalUploadError(
            "progress.json total_dataset_rows must be a nonnegative integer"
        )
    rows = _read_index(run_dir / "index.jsonl")
    if len(rows) != total_runs_value:
        raise LocalEvalUploadError(
            f"local evaluation is incomplete: index.jsonl has {len(rows)} terminal "
            f"results but progress.json expects {total_runs_value}"
        )
    _assert_index_matches_progress(
        rows,
        sampled_rows=int(sampled_rows),
        total_dataset_rows=int(total_dataset_rows),
        n=n_value,
        total_runs=total_runs_value,
    )

    metrics = _json_object(run_dir / "metrics.json", where="metrics.json")
    identity = _object(metrics.get("eval_run"), where="metrics.json eval_run")
    summary = _object(metrics.get("summary"), where="metrics.json summary")
    if identity.get("id") != local_run_id or identity.get("name") != run_name:
        raise LocalEvalUploadError(
            "metrics.json eval_run identity does not match manifest.json"
        )
    if identity.get("status") != "finished":
        raise LocalEvalUploadError(
            "local evaluation is not completed: metrics.json status must be finished"
        )
    started_at = _string(
        identity.get("started_at"), where="metrics.json eval_run.started_at"
    )
    completed_at = _string(
        identity.get("completed_at"), where="metrics.json eval_run.completed_at"
    )
    dataset_name = _string(
        identity.get("dataset_name"), where="metrics.json eval_run.dataset_name"
    )
    if identity.get("model_name") != model_path:
        raise LocalEvalUploadError(
            "metrics.json eval_run.model_name does not match manifest.json"
        )
    if identity.get("rollout_name") != rollout_name:
        raise LocalEvalUploadError(
            "metrics.json eval_run.rollout_name does not match manifest.json"
        )
    pass_threshold = summary.get("pass_threshold")
    if not _is_finite_number(pass_threshold):
        raise LocalEvalUploadError(
            "metrics.json summary.pass_threshold must be a finite number"
        )

    selected: dict[str, Path] = {
        "index.jsonl": run_dir / "index.jsonl",
        "progress.json": run_dir / "progress.json",
    }
    logs_path = run_dir / "logs.txt"
    if not logs_path.is_symlink() and logs_path.is_file():
        selected["logs.txt"] = logs_path
    for row in rows:
        trajectory = _trajectory_path(run_dir, row)
        if trajectory is not None:
            selected[trajectory[0]] = trajectory[1]
        for relative, path in _artifact_paths(run_dir, row["rollout_id"]):
            selected[relative] = path

    files = tuple(
        _hash_file(selected[relative], relative=relative)
        for relative in sorted(selected)
    )
    provenance_source = manifest.get("provenance")
    if provenance_source is not None and not isinstance(provenance_source, dict):
        raise LocalEvalUploadError(
            "manifest.json provenance must be a JSON object when present"
        )
    provenance_object = provenance_source if isinstance(provenance_source, dict) else {}
    provenance: dict[str, Any] = {}
    for key in _PROVENANCE_KEYS:
        value = provenance_object.get(key)
        if value is None:
            continue
        if key == "git_dirty":
            if not isinstance(value, bool):
                raise LocalEvalUploadError(
                    "manifest.json provenance.git_dirty must be a boolean"
                )
        elif not isinstance(value, str) or not value:
            raise LocalEvalUploadError(
                f"manifest.json provenance.{key} must be a non-empty string"
            )
        provenance[key] = value
    rollout_config: dict[str, Any] = {
        "name": rollout_name,
        "source_digest": rollout_source_digest,
    }
    if "config_branch" in provenance:
        rollout_config["branch"] = provenance["config_branch"]
    if "config_commit_sha" in provenance:
        rollout_config["commit_sha"] = provenance["config_commit_sha"]
    return EvalUploadPlan(
        run_dir=run_dir,
        local_run_id=local_run_id,
        manifest_digest=manifest_digest,
        run={
            "name": run_name,
            "started_at": started_at,
            "completed_at": completed_at,
            "experiment_config": {
                "rollout": rollout_config,
                "entrypoint": entrypoint,
                "model_path": model_path,
                "dataset": {
                    "name": dataset_name,
                    "sha256": dataset_sha256,
                    "selected_source_rows": selected_source_rows,
                },
            },
            "evaluation_config": {
                "n": n_value,
                "pass_threshold": float(pass_threshold),
            },
        },
        schema_versions=dict(schema_versions),
        provenance=provenance,
        files=files,
    )
