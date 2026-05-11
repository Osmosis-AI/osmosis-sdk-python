"""Eval cache infrastructure for result storage and resume.

Provides task ID computation, fingerprinting, cache file management,
atomic writes, file locking, and periodic flush control.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import filelock
import xxhash

logger: logging.Logger = logging.getLogger(__name__)

CONTROLLER_PROTOCOL_VERSION = "eval-controller-v1"
_CACHE_VERSION = 3

_ROLLOUT_FINGERPRINT_SUFFIXES = {
    ".py",
    ".toml",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
}
_ROLLOUT_FINGERPRINT_SKIP_DIRS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    "dist",
    "log",
    "logs",
    "venv",
}


# ============================================================
# Deterministic JSON serialization
# ============================================================


def _deterministic_json(obj: object) -> bytes:
    """Serialize to JSON with deterministic float representation.

    Uses Python 3's repr() for float serialization, which guarantees
    round-trip fidelity. Special float values are handled:
    - NaN and Inf raise ValueError
    - -0.0 is normalized to 0.0
    """

    def _normalize(v: object) -> object:
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(
                    f"Non-finite float {v!r} in eval config. "
                    f"Float values must be finite (NaN and Inf are not allowed)."
                )
            if v == 0.0:
                return 0.0  # normalize -0.0 to 0.0
            return v
        if isinstance(v, dict):
            return {k: _normalize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_normalize(item) for item in v]
        return v

    return json.dumps(_normalize(obj), sort_keys=True).encode()


# ============================================================
# Task ID computation
# ============================================================


def compute_task_id(
    *,
    config: dict[str, Any],
    rollout_fingerprint: str,
    dataset_fingerprint: str,
    entrypoint: str,
    controller_protocol_version: str = CONTROLLER_PROTOCOL_VERSION,
) -> tuple[str, str]:
    """Compute (task_id, config_hash) from effective eval configuration.

    Returns a (task_id, config_hash) tuple where config_hash is the full
    xxh3_128 hex digest and task_id is its first 12 characters.

    Hash inputs:
    - config: Dict containing EvalConfig fields that affect result semantics
    - rollout_fingerprint: Hash of rollout filesystem content
    - dataset_fingerprint: Hash of dataset file content
    - entrypoint: Rollout entrypoint path relative to rollout directory
    - controller_protocol_version: Eval controller protocol/cache identity version
    """
    payload = _deterministic_json(
        {
            "config": config,
            "rollout": rollout_fingerprint,
            "dataset": dataset_fingerprint,
            "entrypoint": entrypoint,
            "controller_protocol_version": controller_protocol_version,
        }
    )
    config_hash = xxhash.xxh3_128_hexdigest(payload)
    return config_hash[:12], config_hash


# ============================================================
# Fingerprint computation
# ============================================================


def compute_dataset_fingerprint(path: str | Path) -> str:
    """Compute xxh3_128 fingerprint of dataset file via streaming hash.

    Reads file in 128KB chunks for memory efficiency.

    Raises:
        FileNotFoundError: If the file does not exist. Callers are expected
            to handle this.
    """
    path = Path(path)
    hasher = xxhash.xxh3_128()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(128 * 1024)  # 128KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_rollout_filesystem_fingerprint(
    rollout_dir: str | Path,
    *,
    entrypoint: str,
) -> str:
    """Hash behavior-affecting rollout files for controller-backed eval cache."""
    root = Path(rollout_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Rollout directory not found: {root}")

    entrypoint_path = (root / entrypoint).resolve()
    try:
        entrypoint_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Entrypoint must stay inside rollout directory: {entrypoint}"
        ) from exc
    if not entrypoint_path.is_file():
        raise FileNotFoundError(f"Entrypoint file not found: {entrypoint_path}")

    hasher = xxhash.xxh3_128()
    hasher.update(b"protocol\0")
    protocol = CONTROLLER_PROTOCOL_VERSION.encode()
    hasher.update(len(protocol).to_bytes(8, "big"))
    hasher.update(protocol)

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _ROLLOUT_FINGERPRINT_SKIP_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            try:
                target = path.resolve()
                target.relative_to(root)
            except (OSError, ValueError):
                continue
        if not path.is_file():
            continue
        if path.suffix not in _ROLLOUT_FINGERPRINT_SUFFIXES:
            continue
        rel_bytes = str(rel).encode()
        hasher.update(b"file\0")
        hasher.update(len(rel_bytes).to_bytes(8, "big"))
        hasher.update(rel_bytes)
        hasher.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as file:
            while chunk := file.read(128 * 1024):
                hasher.update(chunk)

    return hasher.hexdigest()


def _resolve_source_file(mod: object) -> Path | None:
    """Resolve the .py source file for a module, handling .pyc and __pycache__."""
    try:
        source_file = inspect.getfile(mod)  # type: ignore[arg-type]
    except (TypeError, OSError):
        return None
    p = Path(source_file)
    if p.suffix in (".pyc", ".pyo"):
        py_path = p.with_suffix(".py")
        if py_path.exists():
            return py_path
        if "__pycache__" in p.parts:
            stem = p.stem.split(".")[0]
            parent_py = p.parent.parent / f"{stem}.py"
            if parent_py.exists():
                return parent_py
        return None
    return p


def compute_eval_fns_fingerprint(eval_fn_paths: list[str]) -> str | None:
    """Hash the source files of all eval functions.

    Deduplicates by file path, sorts for determinism.
    Returns None if any source file cannot be located.
    """
    source_files: dict[str, Path] = {}
    for fn_path in eval_fn_paths:
        module_name, _, _ = fn_path.partition(":")
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            return None
        source_file = _resolve_source_file(mod)
        if source_file is None:
            return None
        source_files[str(source_file.resolve())] = source_file
    if not source_files:
        return None
    hasher = xxhash.xxh3_128()
    for path_str in sorted(source_files.keys()):
        hasher.update(path_str.encode())
        hasher.update(source_files[path_str].read_bytes())
    return hasher.hexdigest()


# ============================================================
# Path resolution and name sanitization
# ============================================================


def _get_cache_root() -> Path:
    """Resolve eval cache root directory.

    Eval caches are stored under the active project's .osmosis directory.
    """
    from osmosis_ai.platform.cli.project_contract import resolve_project_root_from_cwd

    project_root = resolve_project_root_from_cwd()
    return (project_root / ".osmosis" / "cache" / "eval").resolve()


_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM0",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT0",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def sanitize_path_part(s: str, max_len: int = 60) -> str:
    """Sanitize a string for safe use as a directory name.

    - Unicode NFD normalization, strips combining marks (accents)
    - Preserves CJK characters (Unicode word characters)
    - Replaces non-word/non-dash chars with dashes
    - Handles Windows reserved device names
    - Truncation with 4-char hash suffix to avoid collisions
    - Falls back to hash-based name if sanitization produces empty string
    """
    normalized = unicodedata.normalize("NFD", s)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    clean = re.sub(r"[^\w.\-]", "-", normalized)
    clean = re.sub(r"-+", "-", clean).strip("-")
    clean = re.sub(r"\.{2,}", ".", clean).strip(".")
    if not clean:
        return f"eval-{xxhash.xxh3_128_hexdigest(s.encode())[:8]}"
    result = clean.lower()
    if result.upper() in _WINDOWS_RESERVED_NAMES:
        result = f"_{result}"
    if len(result) > max_len:
        suffix = xxhash.xxh3_128_hexdigest(s.encode())[:4]
        result = f"{result[: max_len - 5]}_{suffix}"
    return result


# ============================================================
# Atomic write + file locking
# ============================================================


def _fsync(fd: int) -> None:
    """Flush data to disk. Uses F_FULLFSYNC on macOS for true durability."""
    if sys.platform == "darwin":
        try:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except (ImportError, OSError):
            pass
    os.fsync(fd)


def _replace_with_retry(src: str, dst: Path, max_retries: int = 3) -> None:
    """Atomic rename with retry for Windows antivirus/indexer locks."""
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt >= max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


_WRITE_FAILURE_WARN_THRESHOLD = 3
_consecutive_write_failures = 0


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via NamedTemporaryFile + os.replace().

    Tracks consecutive failures via module-level counter, warns to stderr after 3.
    """
    global _consecutive_write_failures
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            prefix=".cache-",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            _fsync(f.fileno())
        _replace_with_retry(tmp_path, path)
        _consecutive_write_failures = 0
    except BaseException as exc:
        if tmp_path is not None:
            with suppress(OSError):
                os.unlink(tmp_path)
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            _consecutive_write_failures += 1
            if _consecutive_write_failures >= _WRITE_FAILURE_WARN_THRESHOLD:
                logger.warning(
                    "Cache write has failed %d consecutive times (%s: %s). "
                    "Eval results are held in memory but NOT persisted to disk. "
                    "If the process exits, up to %d runs of scores may be lost. "
                    "Check available disk space and permissions for: %s",
                    _consecutive_write_failures,
                    type(exc).__name__,
                    exc,
                    _consecutive_write_failures * 50,
                    path.parent,
                )
        raise


def _get_lock_timeout() -> int:
    """Resolve lock timeout in seconds. Default 30s, override via env var."""
    env_val = os.environ.get("OSMOSIS_EVAL_LOCK_TIMEOUT")
    if env_val is not None:
        try:
            timeout = int(env_val)
            if timeout <= 0:
                raise ValueError
            return timeout
        except ValueError:
            logger.warning(
                "Invalid OSMOSIS_EVAL_LOCK_TIMEOUT=%r, using default 30s",
                env_val,
            )
    return 30


# ============================================================
# CacheConfig dataclass
# ============================================================


@dataclass
class CacheConfig:
    """Encapsulates all configuration needed for cache operations."""

    task_id: str
    config_hash: str
    model: str
    dataset_path: str
    config: dict[str, Any]  # Full config dict for storage in cache file
    total_rows: int


# ============================================================
# Dataset integrity checking
# ============================================================


class DatasetStatus(enum.Enum):
    VALID = "valid"
    MODIFIED = "modified"
    DELETED = "deleted"
    INACCESSIBLE = "inaccessible"


_DATASET_CHECK_INTERVAL_RUNS = 100
_DATASET_CHECK_INTERVAL_SECS = 300.0


class DatasetIntegrityChecker:
    """Periodic dataset fingerprint validation during eval."""

    def __init__(self, dataset_path: Path, expected_fingerprint: str):
        self.dataset_path = dataset_path
        self.expected_fingerprint = expected_fingerprint
        self._runs_since_check = 0
        self._last_check_time = time.monotonic()

    def maybe_check(self) -> DatasetStatus:
        self._runs_since_check += 1
        now = time.monotonic()
        if (
            self._runs_since_check < _DATASET_CHECK_INTERVAL_RUNS
            and now - self._last_check_time < _DATASET_CHECK_INTERVAL_SECS
        ):
            return DatasetStatus.VALID
        self._runs_since_check = 0
        self._last_check_time = now
        try:
            current_fp = compute_dataset_fingerprint(self.dataset_path)
        except FileNotFoundError:
            return DatasetStatus.DELETED
        except OSError:
            return DatasetStatus.INACCESSIBLE
        if current_fp != self.expected_fingerprint:
            return DatasetStatus.MODIFIED
        return DatasetStatus.VALID


# ============================================================
# Cache flush controller
# ============================================================


class CacheFlushController:
    """Controls periodic cache flushing based on accumulated runs and elapsed time."""

    def __init__(
        self,
        cache_path: Path,
        cache_data: dict[str, Any],
        flush_interval_runs: int = 50,
        flush_interval_secs: float = 60.0,
        prior_runs_count: int = 0,
    ):
        self.cache_path = cache_path
        self.cache_data = cache_data
        self.flush_interval_runs = flush_interval_runs
        self.flush_interval_secs = flush_interval_secs
        self._runs_since_flush = 0
        self._last_flush_time = time.monotonic()
        self._prior_runs_count = prior_runs_count

    def maybe_flush(self, runs_completed: int = 1) -> None:
        self._runs_since_flush += runs_completed
        now = time.monotonic()
        if (
            self._runs_since_flush >= self.flush_interval_runs
            or now - self._last_flush_time >= self.flush_interval_secs
        ):
            self._do_flush()

    def force_flush(self) -> None:
        self._do_flush()

    def _do_flush(self) -> None:
        # NOTE: Caller must hold the cache file lock (via CacheBackend.acquire_lock).
        if self._prior_runs_count > 0:
            try:
                disk_cache = json.loads(self.cache_path.read_text())
                old_runs = disk_cache.get("runs", [])[: self._prior_runs_count]
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read prior runs from cache file %s during flush: %s. "
                    "Prior session runs (%d) will be missing from this write.",
                    self.cache_path,
                    exc,
                    self._prior_runs_count,
                )
                old_runs = []
            merged_data = {
                **self.cache_data,
                "runs": old_runs + self.cache_data["runs"],
            }
            atomic_write_json(self.cache_path, merged_data)
        else:
            atomic_write_json(self.cache_path, self.cache_data)
        self._runs_since_flush = 0
        self._last_flush_time = time.monotonic()


# ============================================================
# Cache backend protocol
# ============================================================


class CacheLock(Protocol):
    """Lock handle returned by acquire_lock."""

    def release(self) -> None: ...


class CacheBackend(Protocol):
    """Storage backend for eval cache."""

    def acquire_lock(
        self, task_id: str, model: str, dataset_path: str
    ) -> CacheLock: ...

    def lookup_or_create(
        self,
        task_id: str,
        config_hash: str,
        fresh: bool,
        config: dict[str, Any],
        total_rows: int,
        model: str | None = None,
        dataset_path: str | None = None,
        dataset_fingerprint: str | None = None,
    ) -> tuple[Path, dict[str, Any], set[tuple[int, int, str | None]]]: ...

    def write_cache(self, cache_path: Path, cache_data: dict[str, Any]) -> None: ...

    def append_run(self, cache_path: Path, run: dict[str, Any]) -> None: ...

    def append_sample(self, samples_path: Path, sample: dict[str, Any]) -> None: ...

    def read_samples(self, samples_path: Path) -> list[dict[str, Any]]: ...

    def delete_cache(self, cache_path: Path) -> None: ...

    def list_caches(self, root: Path | None = None) -> list[dict[str, Any]]: ...


# ============================================================
# Corrupt cache backup
# ============================================================


def _backup_corrupt_cache(cache_path: Path) -> Path | None:
    """Rename a corrupt cache file to a .corrupt.{timestamp} backup."""
    ts = int(time.time())
    backup_path = cache_path.with_name(f"{cache_path.name}.corrupt.{ts}_{os.getpid()}")
    try:
        os.replace(cache_path, backup_path)
    except OSError as e:
        logger.warning(
            "Could not back up corrupt cache to %s: %s. "
            "Attempting to overwrite corrupt file in place.",
            backup_path,
            e,
        )
        with suppress(OSError):
            cache_path.unlink()
        return None
    jsonl_path = cache_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        jsonl_backup = jsonl_path.with_name(
            f"{jsonl_path.name}.corrupt.{ts}_{os.getpid()}"
        )
        try:
            os.replace(jsonl_path, jsonl_backup)
        except OSError as e:
            logger.warning(
                "Could not back up corrupt samples to %s: %s",
                jsonl_backup,
                e,
            )
    return backup_path


# ============================================================
# Build summary
# ============================================================


class RewardStatsDict(TypedDict, total=False):
    """Reward statistics returned by build_summary."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    pass_at_k: dict[int, float]


class BuildSummaryResult(TypedDict, total=False):
    """Return type of build_summary."""

    total_runs: int
    passed: int
    failed: int
    skipped: int
    total_tokens: int
    total_duration_ms: float
    reward_stats: RewardStatsDict | None


def build_summary(
    runs: list[dict[str, Any]],
    pass_threshold: float,
    n_runs: int,
) -> BuildSummaryResult:
    """Compute aggregated statistics from runs list."""
    import statistics

    total_runs = len(runs)
    total_tokens = sum(r.get("tokens", 0) for r in runs)
    total_duration_ms = sum(r.get("duration_ms", 0.0) for r in runs)
    skipped = sum(1 for r in runs if r.get("status") == "skipped")
    scored_runs = [r for r in runs if r.get("status") != "skipped"]

    rewards = [r["reward"] for r in scored_runs if r.get("reward") is not None]

    if not rewards:
        return BuildSummaryResult(
            total_runs=total_runs,
            passed=0,
            failed=len(scored_runs),
            skipped=skipped,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            reward_stats=None,
        )

    mean = sum(rewards) / len(rewards)
    variance = sum((s - mean) ** 2 for s in rewards) / len(rewards)
    std = math.sqrt(variance)
    sorted_rewards = sorted(rewards)
    median = statistics.median(sorted_rewards)

    passed = sum(1 for r in rewards if r >= pass_threshold)

    reward_stats: RewardStatsDict = {
        "mean": mean,
        "median": median,
        "std": std,
        "min": sorted_rewards[0],
        "max": sorted_rewards[-1],
    }

    # pass@k for n_runs > 1
    if n_runs > 1:
        from collections import defaultdict

        from osmosis_ai.eval.evaluation.report import pass_at_k

        rows: dict[tuple[int, str | None], list[dict]] = defaultdict(list)
        for r in scored_runs:
            key = (r["row_index"], r.get("model_tag"))
            rows[key].append(r)

        k_values = []
        p = 1
        while p < n_runs:
            k_values.append(p)
            p *= 2
        if not k_values or k_values[-1] != n_runs:
            k_values.append(n_runs)

        pak: dict[int, float] = {}
        for k in k_values:
            row_pass_at_k: list[float] = []
            for row_runs in rows.values():
                c = sum(
                    1
                    for r in row_runs
                    if r.get("reward") is not None and r["reward"] >= pass_threshold
                )
                n = len(row_runs)
                if n > 0 and k <= n:
                    row_pass_at_k.append(pass_at_k(n, c, k))
            if row_pass_at_k:
                pak[k] = sum(row_pass_at_k) / len(row_pass_at_k)
        if pak:
            reward_stats["pass_at_k"] = pak

    return BuildSummaryResult(
        total_runs=total_runs,
        passed=passed,
        failed=len(scored_runs) - passed,
        skipped=skipped,
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
        reward_stats=reward_stats,
    )


# ============================================================
# JSON file cache backend
# ============================================================


class _FileLock:
    """Wrapper around filelock.FileLock implementing CacheLock protocol."""

    def __init__(self, lock_path: Path, timeout: int):
        self._lock = filelock.FileLock(str(lock_path), timeout=timeout)
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


class JsonFileCacheBackend:
    """JSON + JSONL file-based cache backend (v1 implementation)."""

    def __init__(self, cache_root: Path | None = None):
        self._cache_root = cache_root or _get_cache_root()

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def _resolve_cache_dir(self, model: str, dataset_path: str) -> Path:
        model_dir = sanitize_path_part(model)
        dataset_dir = sanitize_path_part(Path(dataset_path).stem)
        return self._cache_root / model_dir / dataset_dir

    def acquire_lock(self, task_id: str, model: str, dataset_path: str) -> CacheLock:
        cache_dir = self._resolve_cache_dir(model, dataset_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / f"{task_id}.lock"
        timeout = _get_lock_timeout()
        try:
            return _FileLock(lock_path, timeout)
        except filelock.Timeout as err:
            raise TimeoutError(
                f"Another eval with the same config is already running.\n"
                f"       Lock file: {lock_path}\n"
                f"       Timeout: {timeout}s (override with OSMOSIS_EVAL_LOCK_TIMEOUT env var)\n"
                f"       Wait for it to finish, or change a config parameter."
            ) from err

    def lookup_or_create(
        self,
        task_id: str,
        config_hash: str,
        fresh: bool,
        config: dict[str, Any],
        total_rows: int,
        model: str | None = None,
        dataset_path: str | None = None,
        dataset_fingerprint: str | None = None,
    ) -> tuple[Path, dict[str, Any], set[tuple[int, int, str | None]]]:
        # Determine cache dir
        _model = model or config.get("model", "unknown")
        _dataset_path = dataset_path or config.get("dataset_path", "unknown")
        cache_dir = self._resolve_cache_dir(_model, _dataset_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Glob for existing cache
        pattern = f"*_{task_id}.json"
        matches = sorted(
            cache_dir.glob(pattern), reverse=True
        )  # newest first by ts prefix

        # --fresh handling
        if fresh and matches:
            for match in matches:
                try:
                    existing = json.loads(match.read_text())
                    existing_runs = existing.get("runs", [])
                except (json.JSONDecodeError, OSError):
                    existing_runs = []

                if existing_runs:
                    ts = int(time.time())
                    backup_path = match.with_name(f"{match.name}.backup.{ts}")
                    os.replace(match, backup_path)
                    # Also backup JSONL
                    jsonl = match.with_suffix(".jsonl")
                    if jsonl.exists():
                        os.replace(jsonl, jsonl.with_name(f"{jsonl.name}.backup.{ts}"))
                    logger.info(
                        "Backed up previous cache (%d runs) to %s",
                        len(existing_runs),
                        backup_path,
                    )
                else:
                    match.unlink(missing_ok=True)
                    jsonl = match.with_suffix(".jsonl")
                    if jsonl.exists():
                        jsonl.unlink()
            matches = []  # treat as Case A

        if not matches:
            # Case A: No cache exists, create new
            unix_ts = int(time.time())
            cache_path = cache_dir / f"{unix_ts}_{task_id}.json"
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            cache_data = {
                "version": _CACHE_VERSION,
                "task_id": task_id,
                "config_hash": config_hash,
                "status": "in_progress",
                "created_at": now_iso,
                "updated_at": now_iso,
                "config": config,
                "runs": [],
                "summary": None,
            }
            atomic_write_json(cache_path, cache_data)
            return cache_path, cache_data, set()

        # Use most recent match
        cache_path = matches[0]
        if len(matches) > 1:
            logger.warning(
                "Multiple cache files found for task %s, using most recent: %s",
                task_id,
                cache_path.name,
            )

        # Parse cache file
        try:
            cache_data = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt file
            backup = _backup_corrupt_cache(cache_path)
            if backup:
                logger.warning(
                    "Cache file corrupt. Backed up to %s. Starting fresh.", backup
                )
            else:
                logger.warning(
                    "Cache file corrupt. Could not create backup; starting fresh."
                )
            # Create new (Case A)
            unix_ts = int(time.time())
            cache_path = cache_dir / f"{unix_ts}_{task_id}.json"
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            cache_data = {
                "version": _CACHE_VERSION,
                "task_id": task_id,
                "config_hash": config_hash,
                "status": "in_progress",
                "created_at": now_iso,
                "updated_at": now_iso,
                "config": config,
                "runs": [],
                "summary": None,
            }
            atomic_write_json(cache_path, cache_data)
            return cache_path, cache_data, set()

        # Version check
        file_version = cache_data.get("version", 0)
        if file_version > _CACHE_VERSION:
            raise RuntimeError(
                f"Cache file created by a newer version of osmosis (v{file_version}). "
                f"Upgrade osmosis or use --fresh."
            )

        if file_version < _CACHE_VERSION:
            logger.warning(
                "Cache file uses older schema (v%d, current v%d). Starting fresh.",
                file_version,
                _CACHE_VERSION,
            )
            _backup_corrupt_cache(cache_path)
            unix_ts = int(time.time())
            cache_path = cache_dir / f"{unix_ts}_{task_id}.json"
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            cache_data = {
                "version": _CACHE_VERSION,
                "task_id": task_id,
                "config_hash": config_hash,
                "status": "in_progress",
                "created_at": now_iso,
                "updated_at": now_iso,
                "config": config,
                "runs": [],
                "summary": None,
            }
            atomic_write_json(cache_path, cache_data)
            return cache_path, cache_data, set()

        status = cache_data.get("status", "in_progress")

        if status == "completed":
            # Case C: Already completed
            completed_runs = {
                (r["row_index"], r["run_index"], r.get("model_tag"))
                for r in cache_data.get("runs", [])
            }
            return cache_path, cache_data, completed_runs

        # Case B: in_progress — verify config_hash
        file_hash = cache_data.get("config_hash", "")
        if file_hash != config_hash:
            raise RuntimeError(
                f"Cache file belongs to a different eval configuration\n"
                f"  that happens to share the same short ID (hash collision).\n"
                f"  Cached config: {file_hash[:16]}...\n"
                f"  Current config: {config_hash[:16]}...\n"
                f"  This is extremely rare. Please re-run with a slightly\n"
                f"  different config (e.g. add --temperature 0.7), or manually\n"
                f"  delete: {cache_path}"
            )

        # Case B: verify dataset fingerprint hasn't changed
        if dataset_fingerprint is not None:
            cached_fp = cache_data.get("config", {}).get("dataset_fingerprint")
            if cached_fp and dataset_fingerprint != cached_fp:
                raise RuntimeError(
                    f"Dataset file has changed since cache was created.\n"
                    f"  Cached: {cached_fp[:16]}... | Current: {dataset_fingerprint[:16]}...\n"
                    f"  Use --fresh to start a new evaluation."
                )

        # Build completed runs set
        completed_runs = {
            (r["row_index"], r["run_index"], r.get("model_tag"))
            for r in cache_data.get("runs", [])
        }
        return cache_path, cache_data, completed_runs

    def write_cache(self, cache_path: Path, cache_data: dict[str, Any]) -> None:
        cache_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(cache_path, cache_data)

    def append_run(self, cache_path: Path, run: dict[str, Any]) -> None:
        pass  # No-op for JSON backend; runs buffered in-memory

    def append_sample(self, samples_path: Path, sample: dict[str, Any]) -> None:
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            line = json.dumps(sample, ensure_ascii=False)
            with open(samples_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except OSError as e:
            logger.warning("Failed to write sample to %s: %s", samples_path, e)

    def read_samples(self, samples_path: Path) -> list[dict[str, Any]]:
        if not samples_path.exists():
            return []
        samples_by_key: dict[tuple[int, int, str | None], dict[str, Any]] = {}
        with open(samples_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = (
                        record["row_index"],
                        record["run_index"],
                        record.get("model_tag"),
                    )
                    samples_by_key[key] = record
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        "Skipping corrupt JSONL line %d in %s", lineno, samples_path
                    )
        return list(samples_by_key.values())

    def delete_cache(self, cache_path: Path) -> None:
        with suppress(OSError):
            cache_path.unlink()
        jsonl_path = cache_path.with_suffix(".jsonl")
        with suppress(OSError):
            jsonl_path.unlink()

    def list_caches(self, root: Path | None = None) -> list[dict[str, Any]]:
        search_root = root or self._cache_root
        if not search_root.exists():
            return []
        results = []
        for json_file in sorted(search_root.rglob("*.json")):
            if (
                json_file.name.startswith(".")
                or ".backup." in json_file.name
                or ".corrupt." in json_file.name
            ):
                continue
            try:
                data = json.loads(json_file.read_text())
                if "version" in data and "task_id" in data:
                    results.append(
                        {
                            "path": str(json_file),
                            "task_id": data.get("task_id"),
                            "status": data.get("status"),
                            "config": data.get("config", {}),
                            "runs_count": len(data.get("runs", [])),
                            "created_at": data.get("created_at"),
                        }
                    )
            except (json.JSONDecodeError, OSError):
                continue
        return results
