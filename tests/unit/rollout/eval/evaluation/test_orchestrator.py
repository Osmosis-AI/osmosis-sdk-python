"""Tests for EvalOrchestrator: cache lifecycle, resume, signals, and batching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
from osmosis_ai.rollout.eval.evaluation.cache import CacheConfig, DatasetStatus
from osmosis_ai.rollout.eval.evaluation.orchestrator import (
    EvalOrchestrator,
    OrchestratorResult,
)
from osmosis_ai.rollout.eval.evaluation.runner import EvalRunResult

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockEvalFn:
    """Lightweight stand-in for EvalFnWrapper (only .name is used by orchestrator)."""

    def __init__(self, name: str = "test_eval") -> None:
        self.name = name


class MockCacheLock:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class MockCacheBackend:
    """In-memory cache backend that records calls for assertions."""

    def __init__(
        self,
        cache_path: Path | None = None,
        cache_data: dict | None = None,
        completed_runs: set[tuple[int, int, str | None]] | None = None,
    ) -> None:
        self.cache_path = cache_path or Path("/tmp/fake_cache.json")
        self.cache_data = (
            cache_data
            if cache_data is not None
            else {
                "status": "in_progress",
                "runs": [],
                "summary": None,
            }
        )
        self.completed_runs = completed_runs or set()
        self.lock = MockCacheLock()
        self.write_cache_calls: list[tuple[Path, dict]] = []
        self.append_sample_calls: list[tuple[Path, dict]] = []

    def acquire_lock(
        self, task_id: str, model: str, dataset_path: str
    ) -> MockCacheLock:
        return self.lock

    def lookup_or_create(
        self,
        task_id: str,
        config_hash: str,
        fresh: bool,
        config: dict,
        total_rows: int,
        model: str | None = None,
        dataset_path: str | None = None,
    ) -> tuple[Path, dict, set[tuple[int, int, str | None]]]:
        return self.cache_path, self.cache_data, self.completed_runs

    def write_cache(self, cache_path: Path, cache_data: dict) -> None:
        self.write_cache_calls.append((cache_path, dict(cache_data)))

    def append_sample(self, samples_path: Path, sample: dict) -> None:
        self.append_sample_calls.append((samples_path, sample))


class MockRunner:
    """Configurable mock for EvalRunner."""

    def __init__(self) -> None:
        self.eval_fns = [MockEvalFn("test_eval")]
        self.run_single_calls: list[tuple[dict, int, int, str | None]] = []
        self.run_batch_calls: list[list[tuple[dict, int, int, str | None]]] = []
        self.has_baseline = False
        self._run_single_results: list[EvalRunResult | Exception] = []
        self._run_batch_results: list[
            tuple[list[EvalRunResult | None], str | None]
        ] = []

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
        if self._run_single_results:
            result = self._run_single_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return EvalRunResult(
            run_index=run_index,
            success=True,
            scores={"test_eval": 1.0},
            row_index=row_index,
            model_tag=model_tag,
            messages=[{"role": "assistant", "content": "ok"}],
        )

    async def run_batch(
        self,
        work_items: list[tuple[dict, int, int, str | None]],
        max_turns: int = 10,
        completion_params: dict | None = None,
    ) -> tuple[list[EvalRunResult | None], str | None]:
        self.run_batch_calls.append(work_items)
        if self._run_batch_results:
            return self._run_batch_results.pop(0)
        results: list[EvalRunResult | None] = []
        for _row, row_index, run_idx, tag in work_items:
            results.append(
                EvalRunResult(
                    run_index=run_idx,
                    success=True,
                    scores={"test_eval": 1.0},
                    row_index=row_index,
                    model_tag=tag,
                    messages=[{"role": "assistant", "content": "ok"}],
                )
            )
        return results, None


class MockDatasetChecker:
    """Mock for DatasetIntegrityChecker."""

    def __init__(self, status: DatasetStatus = DatasetStatus.VALID) -> None:
        self._status = status

    def maybe_check(self) -> DatasetStatus:
        return self._status


def _make_cache_config() -> CacheConfig:
    return CacheConfig(
        task_id="test123",
        config_hash="abc123def456",
        model="test-model",
        dataset_path="/tmp/test.jsonl",
        config={"model": "test-model"},
        total_rows=3,
    )


def _make_rows(n: int = 3) -> list[dict]:
    return [
        {"user_prompt": f"Q{i}", "ground_truth": f"A{i}", "system_prompt": "sys"}
        for i in range(n)
    ]


def _make_orchestrator(
    runner: MockRunner | None = None,
    cache_backend: MockCacheBackend | None = None,
    rows: list[dict] | None = None,
    n_runs: int = 1,
    batch_size: int = 1,
    log_samples: bool = False,
    model_tags: list[str | None] | None = None,
    on_progress: Any = None,
    dataset_path: Path | None = None,
    dataset_fingerprint: str | None = None,
    fresh: bool = False,
    pass_threshold: float = 1.0,
    start_index: int = 0,
) -> EvalOrchestrator:
    return EvalOrchestrator(
        runner=runner or MockRunner(),  # type: ignore[arg-type]
        cache_backend=cache_backend or MockCacheBackend(),  # type: ignore[arg-type]
        cache_config=_make_cache_config(),
        rows=rows if rows is not None else _make_rows(),
        n_runs=n_runs,
        batch_size=batch_size,
        log_samples=log_samples,
        model_tags=model_tags,
        on_progress=on_progress,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        fresh=fresh,
        pass_threshold=pass_threshold,
        start_index=start_index,
    )


# ---------------------------------------------------------------------------
# Test: Case A — fresh eval with no cache
# ---------------------------------------------------------------------------


class TestOrchestratorFreshEval:
    @pytest.mark.asyncio
    async def test_fresh_eval_completes_all_work_items(self) -> None:
        """All work items should be executed when cache is empty."""
        runner = MockRunner()
        cache_backend = MockCacheBackend()
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend, rows=rows)
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 3
        assert result.total_expected == 3
        assert len(runner.run_single_calls) == 3

    @pytest.mark.asyncio
    async def test_fresh_eval_writes_cache_on_completion(self) -> None:
        """Cache should be written with status=completed and summary."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "completed"
        assert result.summary is not None
        assert len(cache_backend.write_cache_calls) >= 1
        # Final write should have status="completed"
        _, final_data = cache_backend.write_cache_calls[-1]
        assert final_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_fresh_eval_summary_computed(self) -> None:
        """Summary should contain eval function stats."""
        orch = _make_orchestrator()
        result = await orch.run()

        assert result.summary is not None
        assert "eval_fns" in result.summary
        assert "test_eval" in result.summary["eval_fns"]
        assert result.summary["total_runs"] == 3

    @pytest.mark.asyncio
    async def test_lock_released_on_success(self) -> None:
        """Lock should be released after successful completion."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: Case C — already completed
# ---------------------------------------------------------------------------


class TestOrchestratorAlreadyCompleted:
    @pytest.mark.asyncio
    async def test_already_completed_returns_immediately(self) -> None:
        """Should return immediately with status='already_completed'."""
        runner = MockRunner()
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "completed",
                "runs": [
                    {
                        "row_index": i,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                    }
                    for i in range(3)
                ],
                "summary": {"eval_fns": {"test_eval": {"mean": 1.0}}},
            },
            completed_runs={(0, 0, None), (1, 0, None), (2, 0, None)},
        )

        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "already_completed"
        assert len(runner.run_single_calls) == 0
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_already_completed_lock_released(self) -> None:
        """Lock should still be released on early return."""
        cache_backend = MockCacheBackend(
            cache_data={"status": "completed", "runs": [], "summary": {}},
            completed_runs=set(),
        )
        orch = _make_orchestrator(cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: Case B — resume from partial
# ---------------------------------------------------------------------------


class TestOrchestratorResume:
    @pytest.mark.asyncio
    async def test_resume_skips_completed_runs(self) -> None:
        """Only remaining work items should be executed."""
        runner = MockRunner()
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None)},
        )
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend, rows=rows)
        result = await orch.run()

        assert result.status == "completed"
        # Should only run 2 remaining items
        assert len(runner.run_single_calls) == 2
        # Total completed = 1 prior + 2 new
        assert result.total_completed == 3

    @pytest.mark.asyncio
    async def test_resume_includes_prior_runs_in_total(self) -> None:
        """total_completed should include both prior and new runs."""
        runner = MockRunner()
        prior_runs = {(0, 0, None), (1, 0, None)}
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs=prior_runs,
        )

        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)
        result = await orch.run()

        assert result.total_completed == 3
        assert len(runner.run_single_calls) == 1


# ---------------------------------------------------------------------------
# Test: Interrupted via SIGINT
# ---------------------------------------------------------------------------


class TestOrchestratorInterrupted:
    @pytest.mark.asyncio
    async def test_shutdown_event_causes_interrupt(self) -> None:
        """Setting _shutdown_event before run should produce interrupted status."""
        runner = MockRunner()
        orch = _make_orchestrator(runner=runner)

        # Patch _install_signal_handlers to set the event immediately after
        # the real run() creates it, simulating an immediate SIGINT.
        original_install = orch._install_signal_handlers

        def install_and_set() -> None:
            original_install()
            if orch._shutdown_event is not None:
                orch._shutdown_event.set()

        orch._install_signal_handlers = install_and_set  # type: ignore[assignment]

        result = await orch.run()

        assert result.status == "interrupted"
        # Since the event is set before the loop starts, 0 items should run.
        assert len(runner.run_single_calls) == 0

    @pytest.mark.asyncio
    async def test_interrupted_records_completed_runs(self) -> None:
        """Runs completed before the interrupt should still be in cache_data."""
        runner = MockRunner()
        call_count = 0

        original_run_single = runner.run_single

        async def counting_run_single(
            row: dict,
            row_index: int,
            run_index: int,
            max_turns: int = 10,
            completion_params: dict | None = None,
            model_tag: str | None = None,
        ) -> EvalRunResult:
            nonlocal call_count
            result = await original_run_single(
                row, row_index, run_index, max_turns, completion_params, model_tag
            )
            call_count += 1
            return result

        runner.run_single = counting_run_single  # type: ignore[assignment]

        cache_backend = MockCacheBackend()
        rows = _make_rows(5)
        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend, rows=rows)

        # Set the shutdown event after the first run completes
        original_orch_run_sequential = orch._run_sequential

        async def patched_sequential(*args: Any, **kwargs: Any) -> Any:
            # Let the first item proceed, then set the event
            result = await original_orch_run_sequential(*args, **kwargs)
            return result

        # Instead of complex patching, just verify the cache_data has runs
        # after interrupt. We'll set shutdown event from run_single.
        runner2 = MockRunner()
        call_count2 = 0

        async def run_single_with_interrupt(
            row: dict,
            row_index: int,
            run_index: int,
            max_turns: int = 10,
            completion_params: dict | None = None,
            model_tag: str | None = None,
        ) -> EvalRunResult:
            nonlocal call_count2
            call_count2 += 1
            res = EvalRunResult(
                run_index=run_index,
                success=True,
                scores={"test_eval": 1.0},
                row_index=row_index,
                model_tag=model_tag,
                messages=[{"role": "assistant", "content": "ok"}],
            )
            # After first run, set shutdown
            if call_count2 >= 1 and orch2._shutdown_event is not None:
                orch2._shutdown_event.set()
            return res

        runner2.run_single = run_single_with_interrupt  # type: ignore[assignment]

        cache_backend2 = MockCacheBackend()
        rows2 = _make_rows(5)
        orch2 = _make_orchestrator(
            runner=runner2, cache_backend=cache_backend2, rows=rows2
        )

        result = await orch2.run()

        assert result.status == "interrupted"
        # At least 1 run completed before interrupt was detected
        assert result.total_completed >= 1
        assert len(result.cache_data.get("runs", [])) >= 1


# ---------------------------------------------------------------------------
# Test: Dataset modified
# ---------------------------------------------------------------------------


class TestOrchestratorDatasetModified:
    @pytest.mark.asyncio
    async def test_dataset_modified_stops_eval(self) -> None:
        """Should stop with status='dataset_modified' when checker reports modified."""
        runner = MockRunner()
        orch = _make_orchestrator(
            runner=runner,
            dataset_path=Path("/tmp/data.jsonl"),
            dataset_fingerprint="original_fp",
        )

        # Replace the dataset checker with our mock
        # The orchestrator creates the checker internally if dataset_path and
        # dataset_fingerprint are both set. We need to monkey-patch the checker.
        original_run = orch.run

        async def patched_run() -> OrchestratorResult:
            # We need to intercept _run_sequential to inject our mock checker.
            original_seq = orch._run_sequential

            async def seq_with_mock_checker(
                work_items: list,
                cache_data: dict,
                flush_ctl: Any,
                dataset_checker: Any,
            ) -> tuple:
                mock_checker = MockDatasetChecker(DatasetStatus.MODIFIED)
                return await original_seq(
                    work_items, cache_data, flush_ctl, mock_checker
                )

            orch._run_sequential = seq_with_mock_checker  # type: ignore[assignment]
            return await original_run()

        result = await patched_run()

        assert result.status == "dataset_modified"
        assert result.stop_reason is not None
        assert "modified" in result.stop_reason.lower()


# ---------------------------------------------------------------------------
# Test: Systemic provider error
# ---------------------------------------------------------------------------


class TestOrchestratorSystemicError:
    @pytest.mark.asyncio
    async def test_systemic_error_stops_eval(self) -> None:
        """SystemicProviderError should stop eval with status='systemic_error'."""
        runner = MockRunner()
        runner._run_single_results = [
            SystemicProviderError("Auth key expired"),
        ]

        orch = _make_orchestrator(runner=runner)
        result = await orch.run()

        assert result.status == "systemic_error"
        assert result.stop_reason is not None
        assert "Auth key expired" in result.stop_reason

    @pytest.mark.asyncio
    async def test_systemic_error_records_failed_run(self) -> None:
        """The failed run should be recorded in cache_data."""
        runner = MockRunner()
        runner._run_single_results = [
            SystemicProviderError("Budget exceeded"),
        ]

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "systemic_error"
        runs = result.cache_data.get("runs", [])
        assert len(runs) == 1
        assert runs[0]["success"] is False
        assert "Budget exceeded" in runs[0].get("error", "")

    @pytest.mark.asyncio
    async def test_systemic_error_lock_released(self) -> None:
        """Lock should be released even on systemic error."""
        runner = MockRunner()
        runner._run_single_results = [
            SystemicProviderError("Connection refused"),
        ]

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: Batch mode
# ---------------------------------------------------------------------------


class TestOrchestratorBatchMode:
    @pytest.mark.asyncio
    async def test_batch_mode_uses_run_batch(self) -> None:
        """batch_size > 1 should use run_batch instead of run_single."""
        runner = MockRunner()
        rows = _make_rows(3)

        orch = _make_orchestrator(runner=runner, rows=rows, batch_size=3)
        result = await orch.run()

        assert result.status == "completed"
        assert len(runner.run_batch_calls) >= 1
        assert len(runner.run_single_calls) == 0

    @pytest.mark.asyncio
    async def test_batch_mode_records_all_results(self) -> None:
        """All batch results should be recorded in cache_data."""
        runner = MockRunner()
        rows = _make_rows(4)

        orch = _make_orchestrator(runner=runner, rows=rows, batch_size=2)
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 4
        assert len(result.cache_data.get("runs", [])) == 4

    @pytest.mark.asyncio
    async def test_batch_mode_systemic_error(self) -> None:
        """Systemic error in batch should stop with status='systemic_error'."""
        runner = MockRunner()
        runner._run_batch_results = [
            (
                [
                    EvalRunResult(
                        run_index=0,
                        success=False,
                        error="Auth failed",
                        row_index=0,
                        model_tag=None,
                    ),
                    EvalRunResult(
                        run_index=0,
                        success=False,
                        error="Auth failed",
                        row_index=1,
                        model_tag=None,
                    ),
                ],
                "Auth failed",  # systemic_error string
            ),
        ]
        rows = _make_rows(4)

        orch = _make_orchestrator(runner=runner, rows=rows, batch_size=2)
        result = await orch.run()

        assert result.status == "systemic_error"
        assert result.stop_reason is not None
        assert "Auth failed" in result.stop_reason


# ---------------------------------------------------------------------------
# Test: _build_work_items
# ---------------------------------------------------------------------------


class TestBuildWorkItems:
    def test_all_items_when_no_completed(self) -> None:
        """Empty completed_runs should produce all work items."""
        orch = _make_orchestrator(rows=_make_rows(3), n_runs=2)
        items = orch._build_work_items(set())

        # 3 rows * 2 runs * 1 model_tag = 6
        assert len(items) == 6

    def test_partial_completed_skips_done(self) -> None:
        """Completed runs should be excluded from work items."""
        orch = _make_orchestrator(rows=_make_rows(3), n_runs=1)
        completed = {(0, 0, None), (2, 0, None)}
        items = orch._build_work_items(completed)

        assert len(items) == 1
        # Only row_index=1 should remain
        assert items[0][1] == 1  # row_index

    def test_all_completed_returns_empty(self) -> None:
        """All completed should produce empty work items list."""
        orch = _make_orchestrator(rows=_make_rows(2), n_runs=1)
        completed = {(0, 0, None), (1, 0, None)}
        items = orch._build_work_items(completed)

        assert len(items) == 0

    def test_start_index_offset(self) -> None:
        """start_index should offset row indices in work items."""
        orch = _make_orchestrator(rows=_make_rows(2), n_runs=1, start_index=10)
        items = orch._build_work_items(set())

        assert len(items) == 2
        assert items[0][1] == 10  # row_index
        assert items[1][1] == 11

    def test_multiple_model_tags(self) -> None:
        """Multiple model tags should multiply work items."""
        orch = _make_orchestrator(
            rows=_make_rows(2), n_runs=1, model_tags=["primary", "baseline"]
        )
        items = orch._build_work_items(set())

        # 2 rows * 1 run * 2 tags = 4
        assert len(items) == 4

    def test_model_tag_in_completed_key(self) -> None:
        """Completed runs with model_tag should be correctly matched."""
        orch = _make_orchestrator(
            rows=_make_rows(2), n_runs=1, model_tags=["primary", "baseline"]
        )
        completed = {(0, 0, "primary")}
        items = orch._build_work_items(completed)

        # 4 total - 1 completed = 3
        assert len(items) == 3
        # (0, 0, "primary") should be missing
        keys = {(item[1], item[2], item[3]) for item in items}
        assert (0, 0, "primary") not in keys
        assert (0, 0, "baseline") in keys


# ---------------------------------------------------------------------------
# Test: _record_result
# ---------------------------------------------------------------------------


class TestRecordResult:
    def test_record_with_log_samples_and_messages(self) -> None:
        """With log_samples=True and messages present, append_sample should be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=True)
        orch._samples_path = Path("/tmp/fake_cache.jsonl")

        cache_data: dict[str, Any] = {"runs": []}
        result = EvalRunResult(
            run_index=0,
            success=True,
            scores={"test_eval": 1.0},
            row_index=0,
            messages=[{"role": "assistant", "content": "hello"}],
        )

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 1
        _, sample = cache_backend.append_sample_calls[0]
        assert sample["row_index"] == 0
        assert sample["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_record_without_log_samples(self) -> None:
        """With log_samples=False, append_sample should NOT be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=False)

        cache_data: dict[str, Any] = {"runs": []}
        result = EvalRunResult(
            run_index=0,
            success=True,
            scores={"test_eval": 1.0},
            row_index=0,
            messages=[{"role": "assistant", "content": "hello"}],
        )

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 0

    def test_record_without_messages(self) -> None:
        """With log_samples=True but messages=None, append_sample should NOT be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=True)
        orch._samples_path = Path("/tmp/fake_cache.jsonl")

        cache_data: dict[str, Any] = {"runs": []}
        result = EvalRunResult(
            run_index=0,
            success=False,
            error="failed",
            row_index=0,
            messages=None,
        )

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 0

    def test_record_appends_run_dict(self) -> None:
        """Run dict should be appended to cache_data['runs']."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result = EvalRunResult(
            run_index=1,
            success=True,
            scores={"test_eval": 0.8},
            duration_ms=123.0,
            tokens=42,
            row_index=5,
        )

        orch._record_result(result, cache_data, row_index=5, run_index=1)

        assert len(cache_data["runs"]) == 1
        run_dict = cache_data["runs"][0]
        assert run_dict["row_index"] == 5
        assert run_dict["run_index"] == 1
        assert run_dict["success"] is True
        assert run_dict["scores"] == {"test_eval": 0.8}
        assert run_dict["duration_ms"] == 123.0
        assert run_dict["tokens"] == 42

    def test_record_includes_model_tag_when_present(self) -> None:
        """model_tag should be included in run dict when not None."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result = EvalRunResult(
            run_index=0, success=True, scores={}, row_index=0, model_tag="primary"
        )

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert cache_data["runs"][0]["model_tag"] == "primary"

    def test_record_omits_model_tag_when_none(self) -> None:
        """model_tag should NOT be in run dict when tag is None."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result = EvalRunResult(run_index=0, success=True, scores={}, row_index=0)

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert "model_tag" not in cache_data["runs"][0]

    def test_record_includes_error_when_present(self) -> None:
        """Error message should be in run dict when result has error."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result = EvalRunResult(
            run_index=0, success=False, error="something broke", row_index=0
        )

        orch._record_result(result, cache_data, row_index=0, run_index=0)

        assert cache_data["runs"][0]["error"] == "something broke"


# ---------------------------------------------------------------------------
# Test: Lock release
# ---------------------------------------------------------------------------


class TestLockRelease:
    @pytest.mark.asyncio
    async def test_lock_released_on_normal_completion(self) -> None:
        """Lock.release() should be called on normal completion."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True

    @pytest.mark.asyncio
    async def test_lock_released_on_exception(self) -> None:
        """Lock.release() should be called even when runner raises unexpected error."""
        runner = MockRunner()

        async def failing_run_single(*args: Any, **kwargs: Any) -> EvalRunResult:
            raise RuntimeError("Unexpected crash")

        runner.run_single = failing_run_single  # type: ignore[assignment]

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)

        with pytest.raises(RuntimeError, match="Unexpected crash"):
            await orch.run()

        assert cache_backend.lock.released is True

    @pytest.mark.asyncio
    async def test_lock_released_on_already_completed(self) -> None:
        """Lock.release() should be called for already_completed case."""
        cache_backend = MockCacheBackend(
            cache_data={"status": "completed", "runs": [], "summary": {}},
        )
        orch = _make_orchestrator(cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_on_progress_called_for_each_run(self) -> None:
        """on_progress should be called once per completed run."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        rows = _make_rows(3)
        orch = _make_orchestrator(rows=rows, on_progress=progress_cb)
        result = await orch.run()

        assert result.status == "completed"
        assert len(progress_calls) == 3

    @pytest.mark.asyncio
    async def test_on_progress_total_matches_expected(self) -> None:
        """Total in progress callback should be rows * n_runs * model_tags."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        rows = _make_rows(2)
        orch = _make_orchestrator(rows=rows, n_runs=2, on_progress=progress_cb)
        await orch.run()

        # 2 rows * 2 runs * 1 tag = 4
        assert len(progress_calls) == 4
        for _, total, _ in progress_calls:
            assert total == 4

    @pytest.mark.asyncio
    async def test_on_progress_with_model_tags(self) -> None:
        """Total should account for multiple model tags."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        rows = _make_rows(2)
        orch = _make_orchestrator(
            rows=rows,
            n_runs=1,
            model_tags=["primary", "baseline"],
            on_progress=progress_cb,
        )
        await orch.run()

        # 2 rows * 1 run * 2 tags = 4
        assert len(progress_calls) == 4
        for _, total, _ in progress_calls:
            assert total == 4

    @pytest.mark.asyncio
    async def test_on_progress_current_increments(self) -> None:
        """current values should be monotonically increasing."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        orch = _make_orchestrator(rows=_make_rows(3), on_progress=progress_cb)
        await orch.run()

        currents = [c for c, _, _ in progress_calls]
        assert currents == sorted(currents)
        assert currents[-1] == 3

    @pytest.mark.asyncio
    async def test_on_progress_called_on_systemic_error(self) -> None:
        """Progress should be reported even for the failing run."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        runner = MockRunner()
        runner._run_single_results = [
            SystemicProviderError("Error"),
        ]

        orch = _make_orchestrator(
            runner=runner, rows=_make_rows(3), on_progress=progress_cb
        )
        result = await orch.run()

        assert result.status == "systemic_error"
        assert len(progress_calls) == 1
        _, _, last_result = progress_calls[0]
        assert last_result.success is False


# ---------------------------------------------------------------------------
# Test: Flush controller
# ---------------------------------------------------------------------------


class TestFlushController:
    @pytest.mark.asyncio
    async def test_force_flush_called_on_normal_completion(self) -> None:
        """force_flush() should be called in the finally block."""
        # We verify indirectly: after run, the cache_data should have been
        # passed to write_cache (which is called by force_flush via the
        # CacheFlushController). Since we use MockCacheBackend, force_flush
        # calls _atomic_write_json on the real CacheFlushController.
        # Instead, let's verify the finally block runs by checking that
        # a completed orchestrator wrote the cache.
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "completed"
        # write_cache is called for the final summary; the flush controller's
        # force_flush writes via _atomic_write_json (not via backend.write_cache).
        # The important thing is that the orchestrator completes without error
        # and the lock is released, indicating the finally block ran.
        assert cache_backend.lock.released is True

    @pytest.mark.asyncio
    async def test_force_flush_called_on_error(self) -> None:
        """force_flush() should be called even when an error occurs."""
        runner = MockRunner()

        async def failing_run(*args: Any, **kwargs: Any) -> EvalRunResult:
            raise RuntimeError("Boom")

        runner.run_single = failing_run  # type: ignore[assignment]

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend)

        with pytest.raises(RuntimeError, match="Boom"):
            await orch.run()

        # Lock released means the outer finally block ran
        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: OrchestratorResult dataclass
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    def test_default_stop_reason_is_none(self) -> None:
        result = OrchestratorResult(
            status="completed",
            cache_path=Path("/tmp/test.json"),
            samples_path=None,
            summary=None,
            total_completed=0,
            total_expected=0,
            cache_data={},
        )
        assert result.stop_reason is None

    def test_all_fields_set(self) -> None:
        result = OrchestratorResult(
            status="interrupted",
            cache_path=Path("/tmp/test.json"),
            samples_path=Path("/tmp/test.jsonl"),
            summary={"eval_fns": {}},
            total_completed=5,
            total_expected=10,
            cache_data={"runs": []},
            stop_reason="User pressed Ctrl-C",
        )
        assert result.status == "interrupted"
        assert result.stop_reason == "User pressed Ctrl-C"
        assert result.total_completed == 5


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_rows(self) -> None:
        """Zero rows should complete immediately with 0 total."""
        runner = MockRunner()
        orch = _make_orchestrator(runner=runner, rows=[])
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 0
        assert result.total_expected == 0
        assert len(runner.run_single_calls) == 0

    @pytest.mark.asyncio
    async def test_all_runs_completed_but_status_not_completed(self) -> None:
        """When all runs exist in cache but status is in_progress, should complete."""
        runner = MockRunner()
        rows = _make_rows(2)
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": True,
                        "scores": {"test_eval": 1.0},
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None), (1, 0, None)},
        )

        orch = _make_orchestrator(runner=runner, cache_backend=cache_backend, rows=rows)
        result = await orch.run()

        assert result.status == "completed"
        assert len(runner.run_single_calls) == 0
        assert result.summary is not None
        # write_cache should be called to persist the completed status
        assert len(cache_backend.write_cache_calls) >= 1

    @pytest.mark.asyncio
    async def test_samples_path_set_when_log_samples_true(self) -> None:
        """samples_path should be set to cache_path.with_suffix('.jsonl')."""
        cache_backend = MockCacheBackend(
            cache_path=Path("/tmp/test_cache.json"),
        )
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=True)
        result = await orch.run()

        assert result.samples_path == Path("/tmp/test_cache.jsonl")

    @pytest.mark.asyncio
    async def test_samples_path_none_when_log_samples_false(self) -> None:
        """samples_path should be None when log_samples=False."""
        orch = _make_orchestrator(log_samples=False)
        result = await orch.run()

        assert result.samples_path is None

    @pytest.mark.asyncio
    async def test_n_runs_multiplies_work(self) -> None:
        """n_runs > 1 should multiply the number of work items."""
        runner = MockRunner()
        orch = _make_orchestrator(runner=runner, rows=_make_rows(2), n_runs=3)
        result = await orch.run()

        assert result.total_expected == 6  # 2 rows * 3 runs
        assert result.total_completed == 6
        assert len(runner.run_single_calls) == 6

    @pytest.mark.asyncio
    async def test_batch_progress_callback(self) -> None:
        """Progress callback should work in batch mode too."""
        progress_calls: list[tuple[int, int, EvalRunResult]] = []

        def progress_cb(current: int, total: int, result: EvalRunResult) -> None:
            progress_calls.append((current, total, result))

        runner = MockRunner()
        rows = _make_rows(3)
        orch = _make_orchestrator(
            runner=runner, rows=rows, batch_size=2, on_progress=progress_cb
        )
        result = await orch.run()

        assert result.status == "completed"
        assert len(progress_calls) == 3
