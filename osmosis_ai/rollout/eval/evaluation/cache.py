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
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Protocol

import filelock
import xxhash

logger: logging.Logger = logging.getLogger(__name__)


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
    model: str,
    base_url: str | None = None,
    baseline_model: str | None = None,
    baseline_base_url: str | None = None,
    module: str,
    dataset: str,
    eval_fns: list[str],
    n_runs: int,
    max_turns: int,
    pass_threshold: float,
    offset: int = 0,
    limit: int | None = None,
    completion_params: dict[str, object] | None = None,
    module_fingerprint: str | None = None,
    dataset_fingerprint: str | None = None,
    eval_fns_fingerprint: str | None = None,
) -> tuple[str, str]:
    """Compute (task_id, config_hash) from full evaluation configuration.

    task_id is the first 12 chars of config_hash (xxh3_128 hex digest).
    None values are filtered before hashing.
    eval_fns are sorted for determinism.
    """
    config: dict[str, object] = {
        "model": model,
        "base_url": base_url,
        "baseline_model": baseline_model,
        "baseline_base_url": baseline_base_url,
        "module": module,
        "module_fingerprint": module_fingerprint,
        "dataset": dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "eval_fns": sorted(eval_fns),
        "eval_fns_fingerprint": eval_fns_fingerprint,
        "n_runs": n_runs,
        "max_turns": max_turns,
        "pass_threshold": pass_threshold,
        "offset": offset,
        "limit": limit,
        "completion_params": completion_params,
    }
    # Filter None values
    config = {k: v for k, v in config.items() if v is not None}
    config_json = _deterministic_json(config)
    config_hash = xxhash.xxh3_128_hexdigest(config_json)
    task_id = config_hash[:12]
    return task_id, config_hash


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


def _hash_file(path: Path) -> str:
    """Hash a single file with xxh3_128."""
    return xxhash.xxh3_128_hexdigest(path.read_bytes())


def _hash_directory_tree(directory: Path) -> str | None:
    """Hash all .py files in a directory tree, sorted for determinism.

    Returns None if no .py files found or any file cannot be read.
    Skips symlinks to avoid infinite loops and external files.
    """
    hasher = xxhash.xxh3_128()
    try:
        py_files = sorted(f for f in directory.rglob("*.py") if not f.is_symlink())
    except OSError:
        return None
    if not py_files:
        return None
    for py_file in py_files:
        try:
            rel_path = py_file.relative_to(directory)
            hasher.update(str(rel_path).encode())
            hasher.update(py_file.read_bytes())
        except (PermissionError, OSError):
            return None
    return hasher.hexdigest()


def compute_module_fingerprint(module_path: str) -> str | None:
    """Hash the agent module's source code.

    - MCP module (mcp:/path/to/dir) -> hash directory tree at that path
    - Package (__init__.py entry) -> hash entire directory tree
    - Single .py file -> hash just that file
    - Returns None if source cannot be located
    """
    if module_path.startswith("mcp:"):
        dir_path = Path(module_path[4:])
        if not dir_path.is_dir():
            return None
        return _hash_directory_tree(dir_path)
    module_name, _, _ = module_path.partition(":")
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    source_file = _resolve_source_file(mod)
    if source_file is None:
        return None
    if source_file.name == "__init__.py":
        return _hash_directory_tree(source_file.parent)
    return _hash_file(source_file)


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
        except Exception:
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

    Priority: OSMOSIS_CACHE_DIR > XDG_CACHE_HOME > ~/.cache/osmosis/eval
    """
    if env := os.environ.get("OSMOSIS_CACHE_DIR"):
        return Path(env).expanduser().resolve() / "eval"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser().resolve() if xdg else Path.home() / ".cache"
    return base / "osmosis" / "eval"


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
    clean = re.sub(r"[^\w-]", "-", normalized)
    clean = re.sub(r"-+", "-", clean).strip("-")
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
            json.dump(data, f, ensure_ascii=False)
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


def build_summary(
    runs: list[dict[str, Any]],
    eval_fn_names: list[str],
    pass_threshold: float,
    n_runs: int,
) -> dict[str, Any]:
    """Compute aggregated statistics from runs list.

    Returns a summary dict with per-eval-fn stats (mean, std, min, max, pass_at_k).
    """
    eval_summaries: dict[str, dict] = {}
    for name in eval_fn_names:
        all_scores = [r["scores"].get(name, 0.0) for r in runs]

        if not all_scores:
            eval_summaries[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            continue

        mean = sum(all_scores) / len(all_scores)
        # Population std (not sample): we compute over all runs, not a sample.
        variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
        std = math.sqrt(variance)

        summary: dict[str, object] = {
            "mean": mean,
            "std": std,
            "min": min(all_scores),
            "max": max(all_scores),
        }

        # pass@k computation for n_runs > 1
        if n_runs > 1:
            # Group runs by (row_index, model_tag)
            rows: dict[tuple[int, str | None], list[dict]] = defaultdict(list)
            for r in runs:
                key = (r["row_index"], r.get("model_tag"))
                rows[key].append(r)

            for k in [1, 3, 5, 10]:
                if k > n_runs:
                    break
                row_pass_at_k: list[float] = []
                for row_runs in rows.values():
                    c = sum(
                        1
                        for r in row_runs
                        if r["scores"].get(name, 0.0) >= pass_threshold
                    )
                    n = len(row_runs)
                    if n > 0 and k <= n:
                        # pass@k formula
                        if c == 0:
                            pak = 0.0
                        elif n <= k or c >= n or n - c < k:
                            pak = 1.0
                        else:
                            pak = 1.0 - comb(n - c, k) / comb(n, k)
                        row_pass_at_k.append(pak)
                if row_pass_at_k:
                    summary[f"pass_at_{k}"] = sum(row_pass_at_k) / len(row_pass_at_k)

        eval_summaries[name] = summary

    return {
        "eval_fns": eval_summaries,
        "total_runs": len(runs),
        "total_tokens": sum(r.get("tokens", 0) for r in runs),
        "total_duration_ms": sum(r.get("duration_ms", 0.0) for r in runs),
    }


# ============================================================
# JSON file cache backend
# ============================================================

_CACHE_VERSION = 1


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
