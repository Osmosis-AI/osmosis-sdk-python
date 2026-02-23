"""Integration tests for the eval cache system (Orchestrator + Cache).

Uses REAL JsonFileCacheBackend with tmp_path for actual file I/O,
while using MockRunner for the agent execution part.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
from osmosis_ai.rollout.eval.evaluation.cache import (
    _CACHE_VERSION,
    CacheConfig,
    JsonFileCacheBackend,
    compute_dataset_fingerprint,
)
from osmosis_ai.rollout.eval.evaluation.orchestrator import (
    EvalOrchestrator,
    OrchestratorResult,
)
from osmosis_ai.rollout.eval.evaluation.runner import EvalRunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEvalFn:
    """Lightweight stand-in for EvalFnWrapper (only .name is used by orchestrator)."""

    def __init__(self, name: str = "test_eval") -> None:
        self.name = name


class MockRunner:
    """Configurable mock for EvalRunner with real file I/O cache backend."""

    def __init__(
        self,
        results: list[EvalRunResult] | None = None,
        fail_after: int | None = None,
        eval_fn_names: list[str] | None = None,
    ) -> None:
        fn_names = eval_fn_names or ["test_eval"]
        self.eval_fns = [MockEvalFn(n) for n in fn_names]
        self.run_single_calls: list[tuple[dict, int, int, str | None]] = []
        self.run_batch_calls: list[list[tuple[dict, int, int, str | None]]] = []
        self.has_baseline = False
        self._results = list(results) if results else []
        self._fail_after = fail_after
        self._call_count = 0

    async def run_single(
        self,
        row: dict,
        row_index: int,
        run_index: int,
        max_turns: int = 10,
        completion_params: dict | None = None,
        model_tag: str | None = None,
    ) -> EvalRunResult:
        self.run_single_calls.append((row, row_index, run_index, model_tag))
        self._call_count += 1

        if self._fail_after is not None and self._call_count > self._fail_after:
            raise SystemicProviderError("Simulated provider failure")

        if self._results:
            result = self._results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return EvalRunResult(
            run_index=run_index,
            success=True,
            scores={"test_eval": 1.0},
            duration_ms=100.0,
            tokens=50,
            row_index=row_index,
            model_tag=model_tag,
            messages=[{"role": "assistant", "content": f"Answer for row {row_index}"}],
        )

    async def run_batch(
        self,
        work_items: list[tuple[dict, int, int, str | None]],
        max_turns: int = 10,
        completion_params: dict | None = None,
    ) -> tuple[list[EvalRunResult | None], str | None]:
        self.run_batch_calls.append(work_items)
        results: list[EvalRunResult | None] = []
        for _row, row_index, run_idx, tag in work_items:
            results.append(
                EvalRunResult(
                    run_index=run_idx,
                    success=True,
                    scores={"test_eval": 1.0},
                    duration_ms=100.0,
                    tokens=50,
                    row_index=row_index,
                    model_tag=tag,
                    messages=[
                        {"role": "assistant", "content": f"Answer for row {row_index}"}
                    ],
                )
            )
        return results, None


def _make_real_cache_backend(cache_root: Path) -> JsonFileCacheBackend:
    """Create a JsonFileCacheBackend using a temp directory."""
    return JsonFileCacheBackend(cache_root=cache_root)


def _make_cache_config(
    task_id: str = "test123abc00",
    config_hash: str = "test123abc00deadbeef12345678abcd",
    model: str = "test-model",
    dataset_path: str = "/tmp/test.jsonl",
    total_rows: int = 3,
) -> CacheConfig:
    """Create a CacheConfig for testing."""
    return CacheConfig(
        task_id=task_id,
        config_hash=config_hash,
        model=model,
        dataset_path=dataset_path,
        config={
            "model": model,
            "dataset_path": dataset_path,
            "n_runs": 1,
            "max_turns": 10,
        },
        total_rows=total_rows,
    )


def _make_rows(n: int = 3) -> list[dict]:
    return [
        {"user_prompt": f"Q{i}", "ground_truth": f"A{i}", "system_prompt": "sys"}
        for i in range(n)
    ]


def _make_orchestrator(
    runner: MockRunner | None = None,
    cache_backend: JsonFileCacheBackend | None = None,
    cache_config: CacheConfig | None = None,
    rows: list[dict] | None = None,
    n_runs: int = 1,
    batch_size: int = 1,
    log_samples: bool = False,
    fresh: bool = False,
    pass_threshold: float = 1.0,
    start_index: int = 0,
    model_tags: list[str | None] | None = None,
    dataset_path: Path | None = None,
    dataset_fingerprint: str | None = None,
    on_progress: Any = None,
) -> EvalOrchestrator:
    cfg = cache_config or _make_cache_config()
    return EvalOrchestrator(
        runner=runner or MockRunner(),  # type: ignore[arg-type]
        cache_backend=cache_backend or JsonFileCacheBackend(),  # type: ignore[arg-type]
        cache_config=cfg,
        rows=rows if rows is not None else _make_rows(),
        n_runs=n_runs,
        batch_size=batch_size,
        log_samples=log_samples,
        fresh=fresh,
        pass_threshold=pass_threshold,
        start_index=start_index,
        model_tags=model_tags,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        on_progress=on_progress,
    )


# ---------------------------------------------------------------------------
# Test 1: Fresh eval with no prior cache
# ---------------------------------------------------------------------------


class TestE2EFreshEval:
    async def test_creates_new_cache_file(self, tmp_path: Path) -> None:
        """Full eval with no prior cache creates a new cache file on disk."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, cache_backend=backend, rows=rows)
        result = await orch.run()

        assert result.status == "completed"
        assert result.cache_path.exists()
        assert result.total_completed == 3
        assert result.total_expected == 3

    async def test_cache_file_has_correct_structure(self, tmp_path: Path) -> None:
        """Cache file has correct JSON structure with all required fields."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, cache_backend=backend, rows=rows)
        result = await orch.run()

        data = json.loads(result.cache_path.read_text())
        assert data["version"] == _CACHE_VERSION
        assert data["task_id"] == "test123abc00"
        assert data["config_hash"] == "test123abc00deadbeef12345678abcd"
        assert data["status"] == "completed"
        assert isinstance(data["runs"], list)
        assert len(data["runs"]) == 3
        assert data["summary"] is not None
        assert "eval_fns" in data["summary"]
        assert "test_eval" in data["summary"]["eval_fns"]
        assert "created_at" in data
        assert "updated_at" in data

    async def test_directory_structure_follows_model_dataset_pattern(
        self, tmp_path: Path
    ) -> None:
        """Cache directory follows model/dataset_stem pattern."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(2)

        orch = _make_orchestrator(runner=runner, cache_backend=backend, rows=rows)
        result = await orch.run()

        # The cache_path should be under eval / <sanitized-model> / <sanitized-dataset>
        rel = result.cache_path.relative_to(tmp_path / "eval")
        parts = rel.parts
        assert len(parts) == 3  # model_dir / dataset_dir / filename.json
        assert parts[0] == "test-model"
        assert parts[2].endswith(".json")

    async def test_orchestrator_result_status(self, tmp_path: Path) -> None:
        """OrchestratorResult has status='completed' and valid summary."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, cache_backend=backend, rows=rows)
        result = await orch.run()

        assert result.status == "completed"
        assert result.summary is not None
        assert result.summary["total_runs"] == 3
        assert result.stop_reason is None


# ---------------------------------------------------------------------------
# Test 2: Resume after interruption
# ---------------------------------------------------------------------------


class TestE2EResumeAfterInterruption:
    async def test_resume_from_partial(self, tmp_path: Path) -> None:
        """Run partial eval, then resume and complete all runs."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(5)
        cfg = _make_cache_config(total_rows=5)

        # First run: fail after 2 successful runs
        runner1 = MockRunner(fail_after=2)
        orch1 = _make_orchestrator(
            runner=runner1,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result1 = await orch1.run()

        assert result1.status == "systemic_error"
        # Should have completed some runs before error
        assert result1.total_completed >= 2

        # Verify cache on disk has partial runs
        disk_data = json.loads(result1.cache_path.read_text())
        assert disk_data["status"] != "completed"
        partial_runs = len(disk_data["runs"])
        assert partial_runs >= 2

        # Second run: resume with a runner that succeeds on all
        runner2 = MockRunner()
        orch2 = _make_orchestrator(
            runner=runner2,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result2 = await orch2.run()

        assert result2.status == "completed"
        assert result2.total_expected == 5

        # Verify the final cache is completed
        final_data = json.loads(result2.cache_path.read_text())
        assert final_data["status"] == "completed"
        assert len(final_data["runs"]) == 5
        assert final_data["summary"] is not None

    async def test_skips_already_completed_runs(self, tmp_path: Path) -> None:
        """On resume, only the remaining runs are executed."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(4)
        cfg = _make_cache_config(total_rows=4)

        # First run: fail after 2
        runner1 = MockRunner(fail_after=2)
        orch1 = _make_orchestrator(
            runner=runner1,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        await orch1.run()

        # Second run: complete the rest
        runner2 = MockRunner()
        orch2 = _make_orchestrator(
            runner=runner2,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result2 = await orch2.run()

        assert result2.status == "completed"
        # runner1 completed 2 runs then failed on the 3rd (which is also recorded).
        # runner2 should only run the remaining items (4 - 3 recorded = 1).
        assert len(runner2.run_single_calls) <= 2


# ---------------------------------------------------------------------------
# Test 3: Already completed cache
# ---------------------------------------------------------------------------


class TestE2EAlreadyCompleted:
    async def test_returns_already_completed(self, tmp_path: Path) -> None:
        """Completed cache returns immediately with status='already_completed'."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(3)
        cfg = _make_cache_config(total_rows=3)

        # First run to completion
        runner1 = MockRunner()
        orch1 = _make_orchestrator(
            runner=runner1,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result1 = await orch1.run()
        assert result1.status == "completed"

        # Second run should return immediately
        runner2 = MockRunner()
        orch2 = _make_orchestrator(
            runner=runner2,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result2 = await orch2.run()

        assert result2.status == "already_completed"
        assert len(runner2.run_single_calls) == 0
        assert result2.summary is not None


# ---------------------------------------------------------------------------
# Test 4: --fresh flag backs up and restarts
# ---------------------------------------------------------------------------


class TestE2EFreshFlag:
    async def test_fresh_backs_up_and_restarts(self, tmp_path: Path) -> None:
        """--fresh backs up the old cache and starts a new eval from scratch."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(3)
        cfg = _make_cache_config(total_rows=3)

        # First run to completion
        runner1 = MockRunner()
        orch1 = _make_orchestrator(
            runner=runner1,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result1 = await orch1.run()
        assert result1.status == "completed"
        old_cache_dir = result1.cache_path.parent

        # Second run with fresh=True
        runner2 = MockRunner()
        orch2 = _make_orchestrator(
            runner=runner2,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
            fresh=True,
        )
        result2 = await orch2.run()

        assert result2.status == "completed"
        # runner2 should have run all 3 items from scratch
        assert len(runner2.run_single_calls) == 3

        # Old cache should have been backed up
        backups = list(old_cache_dir.glob("*.backup.*"))
        assert len(backups) >= 1

    async def test_fresh_with_samples_backup(self, tmp_path: Path) -> None:
        """--fresh also backs up the JSONL samples file."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(2)
        cfg = _make_cache_config(total_rows=2)

        # First run with log_samples
        runner1 = MockRunner()
        orch1 = _make_orchestrator(
            runner=runner1,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
            log_samples=True,
        )
        result1 = await orch1.run()
        assert result1.status == "completed"
        # Ensure samples file was created
        assert result1.samples_path is not None
        assert result1.samples_path.exists()

        # Second run with fresh=True
        runner2 = MockRunner()
        orch2 = _make_orchestrator(
            runner=runner2,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
            fresh=True,
        )
        result2 = await orch2.run()
        assert result2.status == "completed"

        # Backup of JSONL should exist
        jsonl_backups = list(result1.cache_path.parent.glob("*.jsonl.backup.*"))
        assert len(jsonl_backups) >= 1


# ---------------------------------------------------------------------------
# Test 5: JSONL sample logging
# ---------------------------------------------------------------------------


class TestE2ELogSamples:
    async def test_jsonl_file_created(self, tmp_path: Path) -> None:
        """With log_samples=True, a .jsonl file is created alongside .json cache."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            log_samples=True,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.samples_path is not None
        assert result.samples_path.exists()
        assert result.samples_path.suffix == ".jsonl"
        # Should be alongside the JSON file
        assert result.samples_path.parent == result.cache_path.parent
        assert result.samples_path.stem == result.cache_path.stem

    async def test_jsonl_content_format(self, tmp_path: Path) -> None:
        """JSONL content has correct format with row_index, run_index, messages."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            log_samples=True,
        )
        result = await orch.run()

        assert result.samples_path is not None
        lines = result.samples_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert "row_index" in record
            assert "run_index" in record
            assert "messages" in record
            assert record["row_index"] == i
            assert record["run_index"] == 0
            assert isinstance(record["messages"], list)

    async def test_jsonl_not_created_when_disabled(self, tmp_path: Path) -> None:
        """With log_samples=False, no .jsonl file is created."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            log_samples=False,
        )
        result = await orch.run()

        assert result.samples_path is None
        # No jsonl files should exist in the cache directory
        jsonl_files = list(result.cache_path.parent.glob("*.jsonl"))
        assert len(jsonl_files) == 0


# ---------------------------------------------------------------------------
# Test 6: Dataset modified detection
# ---------------------------------------------------------------------------


class TestE2EDatasetModified:
    async def test_dataset_modified_stops_eval(self, tmp_path: Path) -> None:
        """When dataset fingerprint mismatches, eval stops with dataset_modified."""
        backend = _make_real_cache_backend(tmp_path / "eval")

        # Create a real dataset file
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"q": "hello"}\n{"q": "world"}\n')
        compute_dataset_fingerprint(dataset_file)  # verify file is readable

        rows = _make_rows(5)
        cfg = _make_cache_config(total_rows=5)

        # Give a mismatched fingerprint so the integrity check will fail
        # We need the check to trigger, which requires 100 runs or 300s.
        # Instead, modify the file after the first check window.
        # We'll use a runner that modifies the file after a few runs.
        call_count = 0

        class DatasetModifyingRunner(MockRunner):
            async def run_single(
                self,
                row: dict,
                row_index: int,
                run_index: int,
                max_turns: int = 10,
                completion_params: dict | None = None,
                model_tag: str | None = None,
            ) -> EvalRunResult:
                nonlocal call_count
                call_count += 1
                return await super().run_single(
                    row, row_index, run_index, max_turns, completion_params, model_tag
                )

        runner = DatasetModifyingRunner()
        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
            dataset_path=dataset_file,
            dataset_fingerprint="wrong_fingerprint_that_wont_match",
        )

        # Monkey-patch the dataset checker so it triggers on every call
        original_run = orch.run

        async def patched_run() -> OrchestratorResult:
            original_seq = orch._run_sequential

            async def seq_with_forced_check(
                work_items: list,
                cache_data: dict,
                flush_ctl: Any,
                dataset_checker: Any,
                prior_completed_count: int = 0,
            ) -> tuple:
                if dataset_checker is not None:
                    # Force immediate check
                    dataset_checker._runs_since_check = 100
                    dataset_checker._last_check_time = 0.0
                return await original_seq(
                    work_items,
                    cache_data,
                    flush_ctl,
                    dataset_checker,
                    prior_completed_count,
                )

            orch._run_sequential = seq_with_forced_check  # type: ignore[assignment]
            return await original_run()

        result = await patched_run()

        assert result.status == "dataset_modified"
        assert result.stop_reason is not None


# ---------------------------------------------------------------------------
# Test 7: Signal interruption (via shutdown_event)
# ---------------------------------------------------------------------------


class TestE2ESignalInterruption:
    async def test_shutdown_event_interrupts_eval(self, tmp_path: Path) -> None:
        """Setting shutdown_event stops eval with status='interrupted'."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(10)
        cfg = _make_cache_config(total_rows=10)

        call_count = 0

        class InterruptingRunner(MockRunner):
            def __init__(self, orch_ref: list) -> None:
                super().__init__()
                self._orch_ref = orch_ref

            async def run_single(
                self,
                row: dict,
                row_index: int,
                run_index: int,
                max_turns: int = 10,
                completion_params: dict | None = None,
                model_tag: str | None = None,
            ) -> EvalRunResult:
                nonlocal call_count
                call_count += 1
                result = await super().run_single(
                    row, row_index, run_index, max_turns, completion_params, model_tag
                )
                # After 3 runs, set the shutdown event
                if call_count >= 3 and self._orch_ref:
                    orch = self._orch_ref[0]
                    if orch._shutdown_event is not None:
                        orch._shutdown_event.set()
                return result

        orch_ref: list = []
        runner = InterruptingRunner(orch_ref)
        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        orch_ref.append(orch)

        result = await orch.run()

        assert result.status == "interrupted"
        # Should have completed at least 3 runs
        assert result.total_completed >= 3
        # But not all 10
        assert result.total_completed < 10

        # Verify partial progress is saved on disk
        disk_data = json.loads(result.cache_path.read_text())
        assert len(disk_data["runs"]) >= 3

    async def test_interrupted_status_not_completed(self, tmp_path: Path) -> None:
        """Interrupted cache on disk should NOT have status='completed'."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        rows = _make_rows(5)
        cfg = _make_cache_config(total_rows=5)

        class ImmediateInterruptRunner(MockRunner):
            def __init__(self, orch_ref: list) -> None:
                super().__init__()
                self._orch_ref = orch_ref

            async def run_single(self, *args: Any, **kwargs: Any) -> EvalRunResult:
                result = await super().run_single(*args, **kwargs)
                if self._orch_ref:
                    orch = self._orch_ref[0]
                    if orch._shutdown_event is not None:
                        orch._shutdown_event.set()
                return result

        orch_ref: list = []
        runner = ImmediateInterruptRunner(orch_ref)
        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        orch_ref.append(orch)

        result = await orch.run()

        assert result.status == "interrupted"
        disk_data = json.loads(result.cache_path.read_text())
        assert disk_data["status"] != "completed"


# ---------------------------------------------------------------------------
# Test 8: Concurrent lock conflict
# ---------------------------------------------------------------------------


class TestE2EConcurrentLock:
    async def test_lock_conflict_raises_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acquiring lock when already held should raise TimeoutError."""
        # Use a very short lock timeout
        monkeypatch.setenv("OSMOSIS_EVAL_LOCK_TIMEOUT", "1")

        backend = _make_real_cache_backend(tmp_path / "eval")
        cfg = _make_cache_config()

        # Acquire the lock manually
        lock = backend.acquire_lock(cfg.task_id, cfg.model, cfg.dataset_path)
        try:
            # Trying to acquire the same lock should timeout
            with pytest.raises(TimeoutError, match="Another eval"):
                backend.acquire_lock(cfg.task_id, cfg.model, cfg.dataset_path)
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# Test 9: Corrupt cache recovery
# ---------------------------------------------------------------------------


class TestE2ECacheCorruption:
    async def test_corrupt_cache_backed_up_and_fresh_start(
        self, tmp_path: Path
    ) -> None:
        """Corrupt cache file is backed up and eval starts fresh."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        cfg = _make_cache_config()

        # Manually create a corrupt cache file in the expected location
        cache_dir = (tmp_path / "eval") / "test-model" / "test"
        cache_dir.mkdir(parents=True)
        corrupt_path = cache_dir / f"9999999999_{cfg.task_id}.json"
        corrupt_path.write_text("this is not valid json{{{")

        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            cache_config=cfg,
            rows=rows,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 3

        # Verify corrupt backup exists
        corrupt_backups = list(cache_dir.glob("*.corrupt.*"))
        assert len(corrupt_backups) >= 1

        # Verify new cache is valid
        disk_data = json.loads(result.cache_path.read_text())
        assert disk_data["status"] == "completed"
        assert disk_data["version"] == _CACHE_VERSION


# ---------------------------------------------------------------------------
# Test 10: Config hash collision
# ---------------------------------------------------------------------------


class TestE2EConfigHashCollision:
    async def test_different_config_hash_same_task_id_raises(
        self, tmp_path: Path
    ) -> None:
        """Different config_hash with same task_id raises RuntimeError."""
        backend = _make_real_cache_backend(tmp_path / "eval")

        # config_hash check only happens for "in_progress" status.
        # So we create an in_progress cache first.
        cfg3 = _make_cache_config(
            task_id="collide67890",
            config_hash="collide67890aaaaaaaaaaaaaaaaaaa1",
        )
        cfg4 = _make_cache_config(
            task_id="collide67890",
            config_hash="collide67890bbbbbbbbbbbbbbbbbbb2",
        )

        # Create an in_progress cache
        runner3 = MockRunner(fail_after=1)
        orch3 = _make_orchestrator(
            runner=runner3,
            cache_backend=backend,
            cache_config=cfg3,
            rows=_make_rows(5),
        )
        result3 = await orch3.run()
        assert result3.status == "systemic_error"

        # Now try with a different config_hash
        runner4 = MockRunner()
        orch4 = _make_orchestrator(
            runner=runner4,
            cache_backend=backend,
            cache_config=cfg4,
            rows=_make_rows(5),
        )
        with pytest.raises(RuntimeError, match="different eval configuration"):
            await orch4.run()


# ---------------------------------------------------------------------------
# Test 11: Multiple runs (pass@n with n_runs > 1)
# ---------------------------------------------------------------------------


class TestE2EMultipleRuns:
    async def test_n_runs_executes_correct_total(self, tmp_path: Path) -> None:
        """With n_runs=3, each row is evaluated 3 times."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            n_runs=3,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_expected == 9  # 3 rows * 3 runs
        assert result.total_completed == 9
        assert len(runner.run_single_calls) == 9

    async def test_n_runs_summary_correct(self, tmp_path: Path) -> None:
        """Summary is computed correctly for n_runs > 1."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(2)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            n_runs=3,
            pass_threshold=0.5,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.summary is not None
        assert result.summary["total_runs"] == 6  # 2 rows * 3 runs

        # Verify cache on disk
        disk_data = json.loads(result.cache_path.read_text())
        assert len(disk_data["runs"]) == 6

    async def test_n_runs_all_row_run_combinations(self, tmp_path: Path) -> None:
        """All (row_index, run_index) combinations are present in the cache."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(2)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            n_runs=3,
        )
        result = await orch.run()

        disk_data = json.loads(result.cache_path.read_text())
        run_keys = {(r["row_index"], r["run_index"]) for r in disk_data["runs"]}
        expected = {(i, j) for i in range(2) for j in range(3)}
        assert run_keys == expected


# ---------------------------------------------------------------------------
# Test 12: Batch mode
# ---------------------------------------------------------------------------


class TestE2EBatchMode:
    async def test_batch_mode_completes_all(self, tmp_path: Path) -> None:
        """batch_size > 1 runs all work items to completion."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(6)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            batch_size=2,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 6
        assert result.total_expected == 6

    async def test_batch_mode_uses_run_batch(self, tmp_path: Path) -> None:
        """batch_size > 1 should call runner.run_batch, not run_single."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(4)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            batch_size=2,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert len(runner.run_batch_calls) >= 1
        assert len(runner.run_single_calls) == 0

    async def test_batch_mode_cache_correct(self, tmp_path: Path) -> None:
        """Batch mode produces correct cache data on disk."""
        backend = _make_real_cache_backend(tmp_path / "eval")
        runner = MockRunner()
        rows = _make_rows(4)

        orch = _make_orchestrator(
            runner=runner,
            cache_backend=backend,
            rows=rows,
            batch_size=3,
        )
        result = await orch.run()

        assert result.status == "completed"
        disk_data = json.loads(result.cache_path.read_text())
        assert disk_data["status"] == "completed"
        assert len(disk_data["runs"]) == 4
        assert disk_data["summary"] is not None
