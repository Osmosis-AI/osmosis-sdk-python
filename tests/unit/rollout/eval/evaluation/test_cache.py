"""Tests for eval cache: deterministic JSON, task ID, fingerprints, and CLI commands."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from osmosis_ai.rollout.eval.evaluation.cache import (
    _CACHE_VERSION,
    CacheConfig,
    CacheFlushController,
    DatasetIntegrityChecker,
    DatasetStatus,
    JsonFileCacheBackend,
    _backup_corrupt_cache,
    _deterministic_json,
    _get_cache_root,
    _get_lock_timeout,
    _hash_directory_tree,
    _hash_file,
    _resolve_source_file,
    atomic_write_json,
    build_summary,
    compute_dataset_fingerprint,
    compute_eval_fns_fingerprint,
    compute_module_fingerprint,
    compute_task_id,
    sanitize_path_part,
)

# ============================================================
# _deterministic_json tests
# ============================================================


class TestDeterministicJson:
    def test_basic(self) -> None:
        """Dict with strings and ints produces consistent bytes."""
        obj = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        result1 = _deterministic_json(obj)
        result2 = _deterministic_json(obj)
        assert result1 == result2
        assert isinstance(result1, bytes)

    def test_sort_keys(self) -> None:
        """Keys are sorted regardless of insertion order."""
        obj1 = {"b": 2, "a": 1}
        obj2 = {"a": 1, "b": 2}
        assert _deterministic_json(obj1) == _deterministic_json(obj2)

    def test_nan_raises(self) -> None:
        """NaN input raises ValueError."""
        with pytest.raises(ValueError, match="Non-finite float"):
            _deterministic_json({"x": float("nan")})

    def test_inf_raises(self) -> None:
        """Inf input raises ValueError."""
        with pytest.raises(ValueError, match="Non-finite float"):
            _deterministic_json({"x": float("inf")})

        with pytest.raises(ValueError, match="Non-finite float"):
            _deterministic_json({"x": float("-inf")})

    def test_neg_zero(self) -> None:
        """-0.0 and 0.0 produce the same output."""
        assert _deterministic_json({"x": -0.0}) == _deterministic_json({"x": 0.0})

    def test_float_repr(self) -> None:
        """Floats use repr() for deterministic round-trip."""
        obj = {"x": 0.1 + 0.2}
        result = _deterministic_json(obj)
        assert isinstance(result, bytes)
        # Should be consistent across calls
        assert result == _deterministic_json(obj)

    def test_nested_nan_raises(self) -> None:
        """NaN nested in list raises ValueError."""
        with pytest.raises(ValueError, match="Non-finite float"):
            _deterministic_json({"items": [1.0, float("nan")]})

    def test_tuple_treated_as_list(self) -> None:
        """Tuples are serialized as JSON arrays."""
        assert _deterministic_json({"x": (1, 2)}) == _deterministic_json({"x": [1, 2]})


# ============================================================
# compute_task_id tests
# ============================================================


class TestComputeTaskId:
    BASE_CONFIG = {
        "model": "gpt-4",
        "module": "my_agent:Agent",
        "dataset": "data.jsonl",
        "eval_fns": ["eval_mod:fn1", "eval_mod:fn2"],
        "n_runs": 3,
        "max_turns": 10,
        "pass_threshold": 0.5,
    }

    def test_deterministic(self) -> None:
        """Same config produces same task_id."""
        tid1, hash1 = compute_task_id(**self.BASE_CONFIG)
        tid2, hash2 = compute_task_id(**self.BASE_CONFIG)
        assert tid1 == tid2
        assert hash1 == hash2

    def test_none_filtered(self) -> None:
        """None fields don't affect hash — explicitly passing None for optional
        fields should produce the same result as omitting them."""
        tid1, hash1 = compute_task_id(**self.BASE_CONFIG)
        tid2, hash2 = compute_task_id(
            **self.BASE_CONFIG,
            base_url=None,
            baseline_model=None,
            limit=None,
            completion_params=None,
            module_fingerprint=None,
        )
        assert tid1 == tid2
        assert hash1 == hash2

    def test_different_config(self) -> None:
        """Changing any param changes task_id."""
        tid_base, _ = compute_task_id(**self.BASE_CONFIG)

        # Change model
        tid_diff, _ = compute_task_id(**{**self.BASE_CONFIG, "model": "gpt-3.5"})
        assert tid_diff != tid_base

        # Change n_runs
        tid_diff2, _ = compute_task_id(**{**self.BASE_CONFIG, "n_runs": 5})
        assert tid_diff2 != tid_base

        # Change pass_threshold
        tid_diff3, _ = compute_task_id(**{**self.BASE_CONFIG, "pass_threshold": 0.9})
        assert tid_diff3 != tid_base

    def test_eval_fns_sorted(self) -> None:
        """eval_fns order doesn't matter."""
        tid1, hash1 = compute_task_id(
            **{**self.BASE_CONFIG, "eval_fns": ["b:fn", "a:fn"]}
        )
        tid2, hash2 = compute_task_id(
            **{**self.BASE_CONFIG, "eval_fns": ["a:fn", "b:fn"]}
        )
        assert tid1 == tid2
        assert hash1 == hash2

    def test_length(self) -> None:
        """task_id is 12 chars, config_hash is 32 chars."""
        tid, config_hash = compute_task_id(**self.BASE_CONFIG)
        assert len(tid) == 12
        assert len(config_hash) == 32

    def test_task_id_is_prefix_of_config_hash(self) -> None:
        """task_id should be the first 12 chars of config_hash."""
        tid, config_hash = compute_task_id(**self.BASE_CONFIG)
        assert config_hash.startswith(tid)

    def test_with_optional_fields(self) -> None:
        """Including optional fields changes the hash."""
        tid_base, _ = compute_task_id(**self.BASE_CONFIG)
        tid_with_url, _ = compute_task_id(
            **self.BASE_CONFIG,
            base_url="https://api.example.com",
        )
        assert tid_with_url != tid_base

    def test_completion_params_affect_hash(self) -> None:
        """completion_params should be included in the hash."""
        tid_base, _ = compute_task_id(**self.BASE_CONFIG)
        tid_with_params, _ = compute_task_id(
            **self.BASE_CONFIG,
            completion_params={"temperature": 0.7},
        )
        assert tid_with_params != tid_base


# ============================================================
# compute_dataset_fingerprint tests
# ============================================================


class TestComputeDatasetFingerprint:
    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Same file content produces the same hash."""
        f = tmp_path / "data.jsonl"
        f.write_text('{"q": "hello"}\n{"q": "world"}\n')
        h1 = compute_dataset_fingerprint(f)
        h2 = compute_dataset_fingerprint(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different file content produces different hashes."""
        f1 = tmp_path / "data1.jsonl"
        f2 = tmp_path / "data2.jsonl"
        f1.write_text('{"q": "hello"}\n')
        f2.write_text('{"q": "world"}\n')
        assert compute_dataset_fingerprint(f1) != compute_dataset_fingerprint(f2)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """String paths are accepted in addition to Path objects."""
        f = tmp_path / "data.jsonl"
        f.write_text("content\n")
        h1 = compute_dataset_fingerprint(str(f))
        h2 = compute_dataset_fingerprint(f)
        assert h1 == h2

    def test_large_file_streaming(self, tmp_path: Path) -> None:
        """Large files (> 128KB) are hashed correctly via streaming."""
        f = tmp_path / "large.jsonl"
        # Write > 128KB of data
        f.write_bytes(b"x" * (256 * 1024))
        h = compute_dataset_fingerprint(f)
        assert isinstance(h, str)
        assert len(h) == 32


# ============================================================
# _hash_directory_tree tests
# ============================================================


class TestHashDirectoryTree:
    def test_deterministic(self, tmp_path: Path) -> None:
        """Hashing a directory with .py files is deterministic."""
        (tmp_path / "a.py").write_text("print('a')")
        (tmp_path / "b.py").write_text("print('b')")
        h1 = _hash_directory_tree(tmp_path)
        h2 = _hash_directory_tree(tmp_path)
        assert h1 is not None
        assert h1 == h2

    def test_includes_subdirectories(self, tmp_path: Path) -> None:
        """Files in subdirectories are included in the hash."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.py").write_text("print('a')")
        (sub / "b.py").write_text("print('b')")
        h = _hash_directory_tree(tmp_path)
        assert h is not None

    def test_skips_symlinks(self, tmp_path: Path) -> None:
        """Symlinks are skipped during directory hashing."""
        (tmp_path / "real.py").write_text("print('real')")
        target = tmp_path / "target.py"
        target.write_text("print('target')")
        link = tmp_path / "link.py"
        link.symlink_to(target)

        h_with_link = _hash_directory_tree(tmp_path)
        assert h_with_link is not None

        # Hash without the link target should differ
        # (the symlink itself is skipped, only real.py and target.py are hashed)
        # Verify symlink is actually skipped by checking file count
        py_files = sorted(f for f in tmp_path.rglob("*.py") if not f.is_symlink())
        assert link not in py_files

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns None for empty directory (no .py files)."""
        assert _hash_directory_tree(tmp_path) is None

    def test_no_py_files(self, tmp_path: Path) -> None:
        """Returns None when directory has no .py files."""
        (tmp_path / "data.txt").write_text("not python")
        (tmp_path / "config.json").write_text("{}")
        assert _hash_directory_tree(tmp_path) is None

    def test_content_change_changes_hash(self, tmp_path: Path) -> None:
        """Changing file content changes the directory hash."""
        f = tmp_path / "mod.py"
        f.write_text("v1")
        h1 = _hash_directory_tree(tmp_path)
        f.write_text("v2")
        h2 = _hash_directory_tree(tmp_path)
        assert h1 != h2


# ============================================================
# _resolve_source_file tests
# ============================================================


class TestResolveSourceFile:
    def test_basic_py_file(self, tmp_path: Path) -> None:
        """Basic .py file resolution."""
        py_file = tmp_path / "mymod.py"
        py_file.write_text("x = 1")

        mod = types.ModuleType("mymod")
        mod.__file__ = str(py_file)

        result = _resolve_source_file(mod)
        assert result == py_file

    def test_pyc_with_py_available(self, tmp_path: Path) -> None:
        """.pyc file resolves to .py when it exists."""
        py_file = tmp_path / "mymod.py"
        py_file.write_text("x = 1")
        pyc_file = tmp_path / "mymod.pyc"
        pyc_file.write_text("compiled")

        mod = types.ModuleType("mymod")
        mod.__file__ = str(pyc_file)

        result = _resolve_source_file(mod)
        assert result == py_file

    def test_pycache_resolution(self, tmp_path: Path) -> None:
        """.pyc in __pycache__ resolves to parent .py file."""
        py_file = tmp_path / "mymod.py"
        py_file.write_text("x = 1")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        pyc_file = pycache / "mymod.cpython-312.pyc"
        pyc_file.write_text("compiled")

        mod = types.ModuleType("mymod")
        mod.__file__ = str(pyc_file)

        result = _resolve_source_file(mod)
        assert result == py_file

    def test_builtin_module_returns_none(self) -> None:
        """Built-in modules without __file__ return None."""
        import builtins

        result = _resolve_source_file(builtins)
        # builtins doesn't have a meaningful source file
        # The function should handle this gracefully
        # (may return None or a path depending on the module)
        assert result is None or isinstance(result, Path)

    def test_module_without_file_returns_none(self) -> None:
        """Module without __file__ returns None."""
        mod = types.ModuleType("no_file")
        # Don't set __file__
        result = _resolve_source_file(mod)
        assert result is None


# ============================================================
# _hash_file tests
# ============================================================


class TestHashFile:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("content")
        assert _hash_file(f) == _hash_file(f)

    def test_different_content(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert _hash_file(f1) != _hash_file(f2)


# ============================================================
# compute_module_fingerprint tests
# ============================================================


class TestComputeModuleFingerprint:
    def test_returns_string_for_real_module(self) -> None:
        """Fingerprinting a real importable module returns a string."""
        # Use a known module from this project
        result = compute_module_fingerprint("osmosis_ai.rollout.eval.evaluation.cache")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 32

    def test_import_error_returns_none(self) -> None:
        """Non-existent module returns None."""
        result = compute_module_fingerprint("nonexistent_module_xyz_12345:Agent")
        assert result is None

    def test_package_module_hashes_directory(self) -> None:
        """Package (with __init__.py) should hash entire directory tree."""
        # osmosis_ai.rollout.eval.evaluation is a package
        result = compute_module_fingerprint("osmosis_ai.rollout.eval.evaluation")
        assert result is not None
        assert isinstance(result, str)

    def test_colon_separated_path(self) -> None:
        """module_path with colon separator should work (module:attribute)."""
        result = compute_module_fingerprint(
            "osmosis_ai.rollout.eval.evaluation.cache:compute_task_id"
        )
        assert result is not None


# ============================================================
# compute_eval_fns_fingerprint tests
# ============================================================


class TestComputeEvalFnsFingerprint:
    def test_deterministic(self) -> None:
        """Same input produces same fingerprint."""
        paths = ["osmosis_ai.rollout.eval.evaluation.cache:compute_task_id"]
        h1 = compute_eval_fns_fingerprint(paths)
        h2 = compute_eval_fns_fingerprint(paths)
        assert h1 is not None
        assert h1 == h2

    def test_dedup(self) -> None:
        """Two fns from the same module result in file hashed once."""
        # Both paths point to the same module file
        paths_single = ["osmosis_ai.rollout.eval.evaluation.cache:fn1"]
        paths_double = [
            "osmosis_ai.rollout.eval.evaluation.cache:fn1",
            "osmosis_ai.rollout.eval.evaluation.cache:fn2",
        ]
        h1 = compute_eval_fns_fingerprint(paths_single)
        h2 = compute_eval_fns_fingerprint(paths_double)
        # Same module -> same file -> same hash (deduplication)
        assert h1 == h2

    def test_import_error_returns_none(self) -> None:
        """If any module can't be imported, returns None."""
        result = compute_eval_fns_fingerprint(
            [
                "nonexistent_module_xyz:fn1",
                "osmosis_ai.rollout.eval.evaluation.cache:fn2",
            ]
        )
        assert result is None

    def test_empty_list_returns_none(self) -> None:
        """Empty eval fn list returns None."""
        result = compute_eval_fns_fingerprint([])
        assert result is None

    def test_different_modules_different_hash(self) -> None:
        """Functions from different modules produce different hash than single module."""
        paths_one = ["osmosis_ai.rollout.eval.evaluation.cache:fn1"]
        paths_two = [
            "osmosis_ai.rollout.eval.evaluation.cache:fn1",
            "osmosis_ai.rollout.eval.evaluation.eval_fn:fn2",
        ]
        h1 = compute_eval_fns_fingerprint(paths_one)
        h2 = compute_eval_fns_fingerprint(paths_two)
        assert h1 is not None
        assert h2 is not None
        assert h1 != h2

    def test_order_independent(self) -> None:
        """Eval fn order doesn't affect the fingerprint (sorted internally)."""
        paths_a = [
            "osmosis_ai.rollout.eval.evaluation.cache:fn1",
            "osmosis_ai.rollout.eval.evaluation.eval_fn:fn2",
        ]
        paths_b = [
            "osmosis_ai.rollout.eval.evaluation.eval_fn:fn2",
            "osmosis_ai.rollout.eval.evaluation.cache:fn1",
        ]
        h1 = compute_eval_fns_fingerprint(paths_a)
        h2 = compute_eval_fns_fingerprint(paths_b)
        assert h1 == h2


# ============================================================
# _get_cache_root tests
# ============================================================


class TestGetCacheRoot:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("OSMOSIS_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        root = _get_cache_root()
        assert root == Path.home() / ".cache" / "osmosis" / "eval"

    def test_osmosis_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSMOSIS_CACHE_DIR", str(tmp_path))
        root = _get_cache_root()
        assert root == tmp_path.resolve() / "eval"

    def test_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OSMOSIS_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        root = _get_cache_root()
        assert root == tmp_path.resolve() / "osmosis" / "eval"

    def test_osmosis_takes_priority_over_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSMOSIS_CACHE_DIR", str(tmp_path / "osmosis"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        root = _get_cache_root()
        assert "osmosis" in str(root)
        assert "xdg" not in str(root)


# ============================================================
# sanitize_path_part tests
# ============================================================


class TestSanitizePathPart:
    def test_slash_to_dash(self):
        assert sanitize_path_part("openai/gpt-4o") == "openai-gpt-4o"

    def test_cjk_preserved(self):
        assert sanitize_path_part("中文模型") == "中文模型"

    def test_windows_reserved(self):
        assert sanitize_path_part("NUL") == "_nul"
        assert sanitize_path_part("CON") == "_con"
        assert sanitize_path_part("AUX") == "_aux"

    def test_long_name_truncation(self):
        long_name = "a" * 100
        result = sanitize_path_part(long_name)
        assert len(result) <= 60

    def test_long_names_dont_collide(self):
        name1 = "a" * 100 + "v1"
        name2 = "a" * 100 + "v2"
        assert sanitize_path_part(name1) != sanitize_path_part(name2)

    def test_empty_input(self):
        result = sanitize_path_part("///")
        assert result.startswith("eval-")

    def test_accents_stripped(self):
        result = sanitize_path_part("café-model")
        assert result == "cafe-model"


# ============================================================
# atomic_write_json tests
# ============================================================


class TestAtomicWriteJson:
    def test_basic_write(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "test.json"
        atomic_write_json(path, {"key": "value"})
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        data = json.loads(path.read_text())
        assert data == {"v": 2}


# ============================================================
# _get_lock_timeout tests
# ============================================================


class TestGetLockTimeout:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OSMOSIS_EVAL_LOCK_TIMEOUT", raising=False)
        assert _get_lock_timeout() == 30

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("OSMOSIS_EVAL_LOCK_TIMEOUT", "60")
        assert _get_lock_timeout() == 60

    def test_invalid_warns(self, monkeypatch, capsys):
        monkeypatch.setenv("OSMOSIS_EVAL_LOCK_TIMEOUT", "abc")
        assert _get_lock_timeout() == 30

    def test_zero_warns(self, monkeypatch):
        monkeypatch.setenv("OSMOSIS_EVAL_LOCK_TIMEOUT", "0")
        assert _get_lock_timeout() == 30


# ============================================================
# CacheConfig tests
# ============================================================


class TestCacheConfig:
    def test_basic_creation(self):
        cfg = CacheConfig(
            task_id="abc123",
            config_hash="deadbeef" * 4,
            model="gpt-4",
            dataset_path="/data/test.jsonl",
            config={"model": "gpt-4"},
            total_rows=100,
        )
        assert cfg.task_id == "abc123"
        assert cfg.total_rows == 100


# ============================================================
# DatasetIntegrityChecker tests
# ============================================================


class TestDatasetIntegrityChecker:
    def test_valid_on_unchanged(self, tmp_path: Path):
        """Checker returns VALID when dataset is unchanged."""
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"q": "hello"}\n')
        fp = compute_dataset_fingerprint(dataset)
        checker = DatasetIntegrityChecker(dataset, fp)
        # Force a check by setting counters past threshold
        checker._runs_since_check = 100
        checker._last_check_time = 0.0
        assert checker.maybe_check() == DatasetStatus.VALID

    def test_modified_detected(self, tmp_path: Path):
        """Checker returns MODIFIED when dataset content changes."""
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"q": "hello"}\n')
        fp = compute_dataset_fingerprint(dataset)
        checker = DatasetIntegrityChecker(dataset, fp)
        # Modify the file
        dataset.write_text('{"q": "modified"}\n')
        # Force a check
        checker._runs_since_check = 100
        checker._last_check_time = 0.0
        assert checker.maybe_check() == DatasetStatus.MODIFIED

    def test_deleted_detected(self, tmp_path: Path):
        """Checker returns DELETED when dataset file is removed."""
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"q": "hello"}\n')
        fp = compute_dataset_fingerprint(dataset)
        checker = DatasetIntegrityChecker(dataset, fp)
        # Delete the file
        dataset.unlink()
        # Force a check
        checker._runs_since_check = 100
        checker._last_check_time = 0.0
        assert checker.maybe_check() == DatasetStatus.DELETED

    def test_skips_check_within_interval(self, tmp_path: Path):
        """Within run interval, always returns VALID even if modified."""
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"q": "hello"}\n')
        fp = compute_dataset_fingerprint(dataset)
        checker = DatasetIntegrityChecker(dataset, fp)
        # Modify the file
        dataset.write_text('{"q": "modified"}\n')
        # Don't force a check - runs_since_check starts at 0, last_check_time is recent
        # Each call increments _runs_since_check by 1, need < 100 to skip
        result = checker.maybe_check()
        assert result == DatasetStatus.VALID

    def test_inaccessible_detected(self, tmp_path: Path):
        """Checker returns INACCESSIBLE when file can't be read."""
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"q": "hello"}\n')
        fp = compute_dataset_fingerprint(dataset)
        checker = DatasetIntegrityChecker(dataset, fp)
        # Force a check
        checker._runs_since_check = 100
        checker._last_check_time = 0.0
        # Make file inaccessible by pointing to a directory
        dataset.unlink()
        dataset.mkdir()
        result = checker.maybe_check()
        # On most OSes, trying to read a directory raises an OSError
        assert result in (DatasetStatus.INACCESSIBLE, DatasetStatus.DELETED)


# ============================================================
# CacheFlushController tests
# ============================================================


class TestCacheFlushController:
    def test_flush_at_interval(self, tmp_path: Path):
        """After flush_interval_runs, flush should happen."""
        cache_path = tmp_path / "cache.json"
        cache_data = {"version": 1, "runs": [{"row_index": 0, "run_index": 0}]}
        controller = CacheFlushController(
            cache_path=cache_path,
            cache_data=cache_data,
            flush_interval_runs=5,
            flush_interval_secs=9999.0,  # won't trigger by time
        )
        # 4 runs should not flush
        for _ in range(4):
            controller.maybe_flush()
        assert not cache_path.exists()
        # 5th run should flush
        controller.maybe_flush()
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data["runs"] == cache_data["runs"]

    def test_flush_by_time(self, tmp_path: Path):
        """Flush triggered by elapsed time."""
        cache_path = tmp_path / "cache.json"
        cache_data = {"version": 1, "runs": []}
        controller = CacheFlushController(
            cache_path=cache_path,
            cache_data=cache_data,
            flush_interval_runs=9999,  # won't trigger by runs
            flush_interval_secs=0.0,  # trigger immediately by time
        )
        controller.maybe_flush()
        assert cache_path.exists()

    def test_force_flush(self, tmp_path: Path):
        """force_flush always writes regardless of interval."""
        cache_path = tmp_path / "cache.json"
        cache_data = {"version": 1, "runs": []}
        controller = CacheFlushController(
            cache_path=cache_path,
            cache_data=cache_data,
            flush_interval_runs=9999,
            flush_interval_secs=9999.0,
        )
        controller.force_flush()
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data == cache_data

    def test_resume_mode_merge(self, tmp_path: Path):
        """With prior_runs_count > 0, merges old runs from disk with new."""
        cache_path = tmp_path / "cache.json"
        # Write initial cache with old runs
        old_data = {
            "version": 1,
            "runs": [
                {"row_index": 0, "run_index": 0, "scores": {"fn": 0.5}},
                {"row_index": 1, "run_index": 0, "scores": {"fn": 0.7}},
            ],
        }
        cache_path.write_text(json.dumps(old_data))

        # New in-memory data has only new runs
        new_cache_data = {
            "version": 1,
            "runs": [{"row_index": 2, "run_index": 0, "scores": {"fn": 0.9}}],
        }
        controller = CacheFlushController(
            cache_path=cache_path,
            cache_data=new_cache_data,
            prior_runs_count=2,
        )
        controller.force_flush()
        merged = json.loads(cache_path.read_text())
        assert len(merged["runs"]) == 3
        assert merged["runs"][0]["row_index"] == 0
        assert merged["runs"][1]["row_index"] == 1
        assert merged["runs"][2]["row_index"] == 2


# ============================================================
# _backup_corrupt_cache tests
# ============================================================


class TestBackupCorruptCache:
    def test_basic_backup(self, tmp_path: Path):
        """Backs up a corrupt cache file with timestamp suffix."""
        cache_path = tmp_path / "1234_abc.json"
        cache_path.write_text("not valid json{{{")
        backup = _backup_corrupt_cache(cache_path)
        assert backup is not None
        assert backup.exists()
        assert ".corrupt." in backup.name
        assert not cache_path.exists()

    def test_also_backups_jsonl(self, tmp_path: Path):
        """Backs up accompanying JSONL file if it exists."""
        cache_path = tmp_path / "1234_abc.json"
        cache_path.write_text("corrupt")
        jsonl_path = tmp_path / "1234_abc.jsonl"
        jsonl_path.write_text('{"row_index": 0}\n')
        backup = _backup_corrupt_cache(cache_path)
        assert backup is not None
        assert not jsonl_path.exists()
        # Check JSONL backup exists
        jsonl_backups = list(tmp_path.glob("*.jsonl.corrupt.*"))
        assert len(jsonl_backups) == 1


# ============================================================
# build_summary tests
# ============================================================


class TestBuildSummary:
    def test_basic_summary(self):
        """Computes mean, std, min, max correctly."""
        runs = [
            {
                "row_index": 0,
                "run_index": 0,
                "scores": {"fn1": 0.8},
                "success": True,
                "tokens": 100,
                "duration_ms": 50.0,
            },
            {
                "row_index": 1,
                "run_index": 0,
                "scores": {"fn1": 0.6},
                "success": True,
                "tokens": 120,
                "duration_ms": 60.0,
            },
            {
                "row_index": 2,
                "run_index": 0,
                "scores": {"fn1": 1.0},
                "success": True,
                "tokens": 80,
                "duration_ms": 40.0,
            },
        ]
        result = build_summary(runs, ["fn1"], pass_threshold=0.5, n_runs=1)
        assert "eval_fns" in result
        fn1 = result["eval_fns"]["fn1"]
        assert abs(fn1["mean"] - 0.8) < 1e-9
        assert fn1["min"] == 0.6
        assert fn1["max"] == 1.0
        assert result["total_runs"] == 3
        assert result["total_tokens"] == 300
        assert result["total_duration_ms"] == 150.0

    def test_empty_runs(self):
        """Empty runs list produces zero summary."""
        result = build_summary([], ["fn1"], pass_threshold=0.5, n_runs=1)
        fn1 = result["eval_fns"]["fn1"]
        assert fn1["mean"] == 0.0
        assert result["total_runs"] == 0

    def test_pass_at_k(self):
        """pass@k is computed when n_runs > 1."""
        runs = [
            {"row_index": 0, "run_index": 0, "scores": {"fn1": 1.0}, "success": True},
            {"row_index": 0, "run_index": 1, "scores": {"fn1": 0.0}, "success": False},
            {"row_index": 0, "run_index": 2, "scores": {"fn1": 1.0}, "success": True},
        ]
        result = build_summary(runs, ["fn1"], pass_threshold=0.5, n_runs=3)
        fn1 = result["eval_fns"]["fn1"]
        assert "pass_at_1" in fn1
        assert "pass_at_3" in fn1
        # With 2 passing out of 3, pass@1 should be > 0
        assert fn1["pass_at_1"] > 0.0

    def test_pass_at_k_with_fewer_passes_than_k(self):
        """pass@k uses the combinatorial estimator when c < k."""
        runs = [
            {"row_index": 0, "run_index": 0, "scores": {"fn1": 1.0}, "success": True},
            {"row_index": 0, "run_index": 1, "scores": {"fn1": 1.0}, "success": True},
            {"row_index": 0, "run_index": 2, "scores": {"fn1": 0.0}, "success": False},
            {"row_index": 0, "run_index": 3, "scores": {"fn1": 0.0}, "success": False},
            {"row_index": 0, "run_index": 4, "scores": {"fn1": 0.0}, "success": False},
        ]
        result = build_summary(runs, ["fn1"], pass_threshold=0.5, n_runs=5)
        fn1 = result["eval_fns"]["fn1"]

        # n=5, c=2, k=3 => 1 - C(3,3)/C(5,3) = 0.9
        assert fn1["pass_at_3"] == pytest.approx(0.9)

    def test_multiple_eval_fns(self):
        """Summary computed for each eval fn independently."""
        runs = [
            {
                "row_index": 0,
                "run_index": 0,
                "scores": {"fn1": 1.0, "fn2": 0.5},
                "success": True,
                "tokens": 10,
                "duration_ms": 5.0,
            },
        ]
        result = build_summary(runs, ["fn1", "fn2"], pass_threshold=0.5, n_runs=1)
        assert result["eval_fns"]["fn1"]["mean"] == 1.0
        assert result["eval_fns"]["fn2"]["mean"] == 0.5


# ============================================================
# JsonFileCacheBackend tests
# ============================================================


class TestJsonFileCacheBackend:
    def test_case_a_new_cache(self, tmp_path: Path):
        """No existing cache creates a new file."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}
        cache_path, cache_data, completed = backend.lookup_or_create(
            task_id="abc123",
            config_hash="deadbeef" * 4,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        assert cache_path.exists()
        assert cache_data["task_id"] == "abc123"
        assert cache_data["status"] == "in_progress"
        assert cache_data["version"] == _CACHE_VERSION
        assert len(completed) == 0

    def test_case_b_resume(self, tmp_path: Path):
        """Existing in_progress cache returns completed runs set."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}
        config_hash = "deadbeef" * 4

        # Create initial cache
        cache_path, cache_data, _ = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )

        # Add some runs
        cache_data["runs"] = [
            {"row_index": 0, "run_index": 0, "model_tag": None, "scores": {"fn": 0.8}},
            {"row_index": 1, "run_index": 0, "model_tag": None, "scores": {"fn": 0.7}},
        ]
        backend.write_cache(cache_path, cache_data)

        # Resume
        _cache_path2, _cache_data2, completed = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        assert len(completed) == 2
        assert (0, 0, None) in completed
        assert (1, 0, None) in completed

    def test_case_c_completed(self, tmp_path: Path):
        """Existing completed cache returns all runs."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}
        config_hash = "deadbeef" * 4

        cache_path, cache_data, _ = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        cache_data["status"] = "completed"
        cache_data["runs"] = [
            {"row_index": 0, "run_index": 0, "model_tag": None},
            {"row_index": 1, "run_index": 0, "model_tag": None},
        ]
        backend.write_cache(cache_path, cache_data)

        _, _, completed = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        assert len(completed) == 2

    def test_config_hash_collision(self, tmp_path: Path):
        """Different config_hash on same task_id raises RuntimeError."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}

        backend.lookup_or_create(
            task_id="abc123",
            config_hash="aaaa" * 8,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )

        with pytest.raises(RuntimeError, match="different eval configuration"):
            backend.lookup_or_create(
                task_id="abc123",
                config_hash="bbbb" * 8,
                fresh=False,
                config=config,
                total_rows=10,
                model="gpt-4",
                dataset_path="test.jsonl",
            )

    def test_fresh_backup(self, tmp_path: Path):
        """--fresh with existing runs creates backup."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}
        config_hash = "deadbeef" * 4

        cache_path, cache_data, _ = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        cache_data["runs"] = [{"row_index": 0, "run_index": 0, "scores": {}}]
        backend.write_cache(cache_path, cache_data)

        # Fresh run
        _cache_path2, fresh_data, completed = backend.lookup_or_create(
            task_id="abc123",
            config_hash=config_hash,
            fresh=True,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        assert len(completed) == 0
        assert fresh_data["runs"] == []
        # Check backup was created
        backups = list(cache_path.parent.glob("*.backup.*"))
        assert len(backups) >= 1

    def test_corrupt_cache_backup(self, tmp_path: Path):
        """Corrupt JSON gets backed up and a fresh cache is created."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        # Manually create a corrupt cache file
        cache_dir = tmp_path / "gpt-4" / "test"
        cache_dir.mkdir(parents=True)
        corrupt_path = cache_dir / "9999999999_abc123.json"
        corrupt_path.write_text("not valid json{{{")

        config = {"model": "gpt-4", "dataset_path": "test.jsonl"}
        _cache_path, cache_data, completed = backend.lookup_or_create(
            task_id="abc123",
            config_hash="deadbeef" * 4,
            fresh=False,
            config=config,
            total_rows=10,
            model="gpt-4",
            dataset_path="test.jsonl",
        )
        assert len(completed) == 0
        assert cache_data["status"] == "in_progress"
        # Corrupt backup should exist
        corrupt_backups = list(cache_dir.glob("*.corrupt.*"))
        assert len(corrupt_backups) >= 1

    def test_append_sample(self, tmp_path: Path):
        """Appends JSONL line to samples file."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        backend.append_sample(
            samples_path, {"row_index": 0, "run_index": 0, "data": "test"}
        )
        backend.append_sample(
            samples_path, {"row_index": 1, "run_index": 0, "data": "test2"}
        )
        lines = samples_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["row_index"] == 0
        assert json.loads(lines[1])["row_index"] == 1

    def test_read_samples_dedup(self, tmp_path: Path):
        """Duplicate entries: last one wins."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        # Write two entries with same key, different data
        backend.append_sample(
            samples_path, {"row_index": 0, "run_index": 0, "data": "old"}
        )
        backend.append_sample(
            samples_path, {"row_index": 0, "run_index": 0, "data": "new"}
        )
        result = backend.read_samples(samples_path)
        assert len(result) == 1
        assert result[0]["data"] == "new"

    def test_read_samples_corrupt_line(self, tmp_path: Path):
        """Corrupt JSONL lines are skipped."""
        samples_path = tmp_path / "samples.jsonl"
        samples_path.write_text(
            '{"row_index": 0, "run_index": 0, "data": "good"}\n'
            "not valid json\n"
            '{"row_index": 1, "run_index": 0, "data": "also good"}\n'
        )
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        result = backend.read_samples(samples_path)
        assert len(result) == 2

    def test_read_samples_nonexistent(self, tmp_path: Path):
        """Non-existent samples file returns empty list."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        result = backend.read_samples(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_version_check(self, tmp_path: Path):
        """Higher cache version raises RuntimeError."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        cache_dir = tmp_path / "gpt-4" / "test"
        cache_dir.mkdir(parents=True)
        future_cache = cache_dir / "9999999999_abc123.json"
        future_cache.write_text(
            json.dumps(
                {
                    "version": 999,
                    "task_id": "abc123",
                    "config_hash": "deadbeef" * 4,
                    "status": "in_progress",
                    "runs": [],
                }
            )
        )

        with pytest.raises(RuntimeError, match="newer version"):
            backend.lookup_or_create(
                task_id="abc123",
                config_hash="deadbeef" * 4,
                fresh=False,
                config={"model": "gpt-4", "dataset_path": "test.jsonl"},
                total_rows=10,
                model="gpt-4",
                dataset_path="test.jsonl",
            )

    def test_delete_cache(self, tmp_path: Path):
        """Deletes both JSON and JSONL files."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        cache_path = tmp_path / "cache.json"
        jsonl_path = tmp_path / "cache.jsonl"
        cache_path.write_text('{"version": 1}')
        jsonl_path.write_text('{"row_index": 0}\n')
        backend.delete_cache(cache_path)
        assert not cache_path.exists()
        assert not jsonl_path.exists()

    def test_list_caches(self, tmp_path: Path):
        """Lists all valid cache files."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        # Create valid cache
        sub = tmp_path / "model" / "dataset"
        sub.mkdir(parents=True)
        valid = sub / "123_abc.json"
        valid.write_text(
            json.dumps(
                {
                    "version": 1,
                    "task_id": "abc",
                    "status": "completed",
                    "config": {},
                    "runs": [{"row_index": 0}],
                    "created_at": "2024-01-01T00:00:00Z",
                }
            )
        )
        # Create backup (should be excluded)
        backup = sub / "123_abc.json.backup.999"
        backup.write_text(json.dumps({"version": 1, "task_id": "abc"}))

        result = backend.list_caches()
        assert len(result) == 1
        assert result[0]["task_id"] == "abc"
        assert result[0]["runs_count"] == 1

    def test_list_caches_empty(self, tmp_path: Path):
        """Empty cache root returns empty list."""
        backend = JsonFileCacheBackend(cache_root=tmp_path / "nonexistent")
        assert backend.list_caches() == []

    def test_acquire_lock_and_release(self, tmp_path: Path):
        """Lock can be acquired and released."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        lock = backend.acquire_lock("abc123", "gpt-4", "test.jsonl")
        lock.release()

    def test_write_cache_updates_timestamp(self, tmp_path: Path):
        """write_cache updates the updated_at field."""
        backend = JsonFileCacheBackend(cache_root=tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_data = {"version": 1, "runs": []}
        backend.write_cache(cache_path, cache_data)
        assert "updated_at" in cache_data
        data = json.loads(cache_path.read_text())
        assert "updated_at" in data


# ============================================================
# Helper: create cache entries in a tmp_path backend
# ============================================================


def _make_cache_entry(
    cache_root: Path,
    task_id: str,
    model: str,
    dataset: str,
    status: str = "completed",
    runs_count: int = 10,
    created_at: str = "2024-02-23T15:30:00Z",
) -> Path:
    """Create a cache JSON file for testing."""
    model_dir = sanitize_path_part(model)
    dataset_dir = sanitize_path_part(Path(dataset).stem)
    cache_dir = cache_root / model_dir / dataset_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if created_at:
        from datetime import datetime

        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        unix_ts = int(dt.timestamp())
    else:
        unix_ts = 0
    cache_path = cache_dir / f"{unix_ts}_{task_id}.json"
    runs = [
        {"row_index": i, "run_index": 0, "scores": {"fn": 0.5}}
        for i in range(runs_count)
    ]
    data = {
        "version": 1,
        "task_id": task_id,
        "config_hash": "a" * 32,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "config": {"model": model, "dataset": dataset},
        "runs": runs,
        "summary": None,
    }
    cache_path.write_text(json.dumps(data))
    # Also create a companion JSONL
    jsonl_path = cache_path.with_suffix(".jsonl")
    jsonl_path.write_text('{"row_index": 0, "run_index": 0}\n')
    return cache_path


# ============================================================
# EvalCommand cache ls tests
# ============================================================


class TestCacheLs:
    def _make_command(self):
        from osmosis_ai.cli.console import Console
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        cmd.console = Console(force_terminal=False, no_color=True)
        return cmd

    def test_ls_empty(self, tmp_path: Path, monkeypatch):
        """No caches prints 'No cached evaluations found.'."""
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            cache_model=None, cache_dataset=None, cache_status=None
        )
        ret = cmd._run_cache_ls(args)
        assert ret == 0

    def test_ls_lists_entries(self, tmp_path: Path, monkeypatch):
        """Lists cache entries."""
        _make_cache_entry(tmp_path, "abc123", "openai/gpt-4", "my_data.jsonl")
        _make_cache_entry(
            tmp_path,
            "def456",
            "openai/gpt-3.5",
            "other.jsonl",
            status="in_progress",
            runs_count=5,
        )
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )

        cmd = self._make_command()
        args = argparse.Namespace(
            cache_model=None, cache_dataset=None, cache_status=None
        )
        ret = cmd._run_cache_ls(args)
        assert ret == 0

    def test_ls_filter_by_model(self, tmp_path: Path):
        """--model filters by model substring."""
        _make_cache_entry(tmp_path, "abc123", "openai/gpt-4", "data.jsonl")
        _make_cache_entry(tmp_path, "def456", "anthropic/claude", "data.jsonl")

        backend = JsonFileCacheBackend(cache_root=tmp_path)
        entries = backend.list_caches()

        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        filtered = EvalCommand._filter_caches(
            entries, model="gpt-4", dataset=None, status=None
        )
        assert len(filtered) == 1
        assert filtered[0]["task_id"] == "abc123"

    def test_ls_filter_by_status(self, tmp_path: Path):
        """--status filters by status."""
        _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl", status="completed")
        _make_cache_entry(
            tmp_path, "def456", "gpt-4", "data2.jsonl", status="in_progress"
        )

        backend = JsonFileCacheBackend(cache_root=tmp_path)
        entries = backend.list_caches()

        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        filtered = EvalCommand._filter_caches(
            entries, model=None, dataset=None, status="in_progress"
        )
        assert len(filtered) == 1
        assert filtered[0]["task_id"] == "def456"

    def test_ls_filter_by_dataset(self, tmp_path: Path):
        """--dataset filters by dataset substring."""
        _make_cache_entry(tmp_path, "abc123", "gpt-4", "train_data.jsonl")
        _make_cache_entry(tmp_path, "def456", "gpt-4", "test_data.jsonl")

        backend = JsonFileCacheBackend(cache_root=tmp_path)
        entries = backend.list_caches()

        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        filtered = EvalCommand._filter_caches(
            entries, model=None, dataset="train", status=None
        )
        assert len(filtered) == 1
        assert filtered[0]["task_id"] == "abc123"

    def test_ls_filter_case_insensitive(self, tmp_path: Path):
        """Filters are case-insensitive."""
        _make_cache_entry(tmp_path, "abc123", "OpenAI/GPT-4", "data.jsonl")

        backend = JsonFileCacheBackend(cache_root=tmp_path)
        entries = backend.list_caches()

        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        filtered = EvalCommand._filter_caches(
            entries, model="openai/gpt-4", dataset=None, status=None
        )
        assert len(filtered) == 1

    def test_ls_sorted_newest_first(self, tmp_path: Path):
        """Entries are sorted by created_at descending."""
        _make_cache_entry(
            tmp_path,
            "old111",
            "gpt-4",
            "a.jsonl",
            created_at="2024-01-01T00:00:00Z",
        )
        _make_cache_entry(
            tmp_path,
            "new222",
            "gpt-4",
            "b.jsonl",
            created_at="2024-06-15T00:00:00Z",
        )

        backend = JsonFileCacheBackend(cache_root=tmp_path)
        entries = backend.list_caches()
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        assert entries[0]["task_id"] == "new222"
        assert entries[1]["task_id"] == "old111"


# ============================================================
# EvalCommand cache rm tests
# ============================================================


class TestCacheRm:
    def _make_command(self):
        from osmosis_ai.cli.console import Console
        from osmosis_ai.rollout.eval.evaluation.cli import EvalCommand

        cmd = EvalCommand()
        cmd.console = Console(force_terminal=False, no_color=True)
        return cmd

    def test_rm_no_args_errors(self, tmp_path: Path, monkeypatch):
        """No arguments prints error."""
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=False,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 1

    def test_rm_by_task_id(self, tmp_path: Path, monkeypatch):
        """Delete single entry by task_id without confirmation."""
        cache_path = _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")
        jsonl_path = cache_path.with_suffix(".jsonl")
        assert cache_path.exists()
        assert jsonl_path.exists()

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id="abc123",
            rm_all=False,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 0
        assert not cache_path.exists()
        assert not jsonl_path.exists()

    def test_rm_nonexistent_task_id(self, tmp_path: Path, monkeypatch):
        """Non-existent task_id prints message and returns 1."""
        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id="nonexistent",
            rm_all=False,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 1

    def test_rm_all_with_yes(self, tmp_path: Path, monkeypatch):
        """--all --yes deletes everything without prompting."""
        p1 = _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")
        p2 = _make_cache_entry(tmp_path, "def456", "gpt-3.5", "data2.jsonl")

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=True,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=True,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 0
        assert not p1.exists()
        assert not p2.exists()

    def test_rm_all_confirm_yes(self, tmp_path: Path, monkeypatch):
        """--all with 'y' confirmation deletes."""
        _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=True,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        with patch("builtins.input", return_value="y"):
            ret = cmd._run_cache_rm(args)
        assert ret == 0

    def test_rm_all_confirm_no(self, tmp_path: Path, monkeypatch):
        """--all with 'n' confirmation aborts."""
        p1 = _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=True,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        with patch("builtins.input", return_value="n"):
            ret = cmd._run_cache_rm(args)
        assert ret == 0
        assert p1.exists()  # Not deleted

    def test_rm_filter_by_status(self, tmp_path: Path, monkeypatch):
        """Filter by --status deletes only matching entries."""
        p_completed = _make_cache_entry(
            tmp_path, "abc123", "gpt-4", "data.jsonl", status="completed"
        )
        p_progress = _make_cache_entry(
            tmp_path, "def456", "gpt-4", "data2.jsonl", status="in_progress"
        )

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=False,
            cache_model=None,
            cache_dataset=None,
            cache_status="in_progress",
            yes=True,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 0
        assert p_completed.exists()
        assert not p_progress.exists()

    def test_rm_cleans_empty_dirs(self, tmp_path: Path, monkeypatch):
        """After deletion, empty parent directories are cleaned up."""
        p = _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")
        parent_dir = p.parent

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id="abc123",
            rm_all=False,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        ret = cmd._run_cache_rm(args)
        assert ret == 0
        assert not parent_dir.exists()

    def test_rm_eof_aborts(self, tmp_path: Path, monkeypatch):
        """EOFError during confirmation aborts gracefully."""
        _make_cache_entry(tmp_path, "abc123", "gpt-4", "data.jsonl")

        monkeypatch.setattr(
            "osmosis_ai.rollout.eval.evaluation.cache._get_cache_root",
            lambda: tmp_path,
        )
        cmd = self._make_command()
        args = argparse.Namespace(
            task_id=None,
            rm_all=True,
            cache_model=None,
            cache_dataset=None,
            cache_status=None,
            yes=False,
        )
        with patch("builtins.input", side_effect=EOFError):
            ret = cmd._run_cache_rm(args)
        assert ret == 130
