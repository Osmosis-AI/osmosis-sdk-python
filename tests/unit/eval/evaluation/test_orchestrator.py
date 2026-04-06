"""Tests for EvalOrchestrator: cache lifecycle, resume, signals, and batching."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.eval.evaluation.cache import CacheConfig, DatasetStatus
from osmosis_ai.eval.evaluation.orchestrator import (
    EvalOrchestrator,
    OrchestratorResult,
)
from osmosis_ai.rollout_v2.driver import RolloutDriver, RolloutOutcome
from osmosis_ai.rollout_v2.types import RolloutSample, RolloutStatus

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class FakeDriver(RolloutDriver):
    """Configurable fake driver for testing."""

    def __init__(
        self,
        outcomes: list[RolloutOutcome] | None = None,
        max_conc: int = 0,
    ) -> None:
        self._outcomes = outcomes or []
        self._call_count = 0
        self._max_conc = max_conc
        self._concurrent = 0
        self._peak_concurrent = 0

    @property
    def max_concurrency(self) -> int:
        return self._max_conc

    async def run(
        self,
        messages: list[dict[str, Any]],
        label: str | None = None,
        rollout_id: str = "",
    ) -> RolloutOutcome:
        self._concurrent += 1
        self._peak_concurrent = max(self._peak_concurrent, self._concurrent)
        try:
            if self._call_count < len(self._outcomes):
                outcome = self._outcomes[self._call_count]
            else:
                outcome = RolloutOutcome(
                    status=RolloutStatus.SUCCESS,
                    samples={"s1": RolloutSample(id="s1", messages=[], reward=1.0)},
                    duration_ms=100,
                    tokens=50,
                )
            self._call_count += 1
            await asyncio.sleep(0.001)
            return outcome
        finally:
            self._concurrent -= 1


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
        dataset_fingerprint: str | None = None,
    ) -> tuple[Path, dict, set[tuple[int, int, str | None]]]:
        return self.cache_path, self.cache_data, self.completed_runs

    def write_cache(self, cache_path: Path, cache_data: dict) -> None:
        self.write_cache_calls.append((cache_path, dict(cache_data)))

    def append_sample(self, samples_path: Path, sample: dict) -> None:
        self.append_sample_calls.append((samples_path, sample))


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
    drivers: list[tuple[str | None, RolloutDriver]] | None = None,
    cache_backend: MockCacheBackend | None = None,
    rows: list[dict] | None = None,
    n_runs: int = 1,
    batch_size: int = 1,
    log_samples: bool = False,
    on_progress: Any = None,
    dataset_path: Path | None = None,
    dataset_fingerprint: str | None = None,
    fresh: bool = False,
    retry_failed: bool = False,
    pass_threshold: float = 1.0,
    start_index: int = 0,
    has_grader: bool = True,
) -> EvalOrchestrator:
    if drivers is None:
        drivers = [(None, FakeDriver())]
    return EvalOrchestrator(
        drivers=drivers,
        cache_backend=cache_backend or MockCacheBackend(),  # type: ignore[arg-type]
        cache_config=_make_cache_config(),
        rows=rows if rows is not None else _make_rows(),
        n_runs=n_runs,
        batch_size=batch_size,
        log_samples=log_samples,
        on_progress=on_progress,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        fresh=fresh,
        retry_failed=retry_failed,
        pass_threshold=pass_threshold,
        start_index=start_index,
        has_grader=has_grader,
    )


# ---------------------------------------------------------------------------
# Test: Case A — fresh eval with no cache
# ---------------------------------------------------------------------------


class TestOrchestratorFreshEval:
    @pytest.mark.asyncio
    async def test_fresh_eval_completes_all_work_items(self) -> None:
        """All work items should be executed when cache is empty."""
        driver = FakeDriver()
        drivers = [(None, driver)]
        cache_backend = MockCacheBackend()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            drivers=drivers, cache_backend=cache_backend, rows=rows
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 3
        assert result.total_expected == 3
        assert driver._call_count == 3

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
        """Summary should contain reward stats."""
        orch = _make_orchestrator()
        result = await orch.run()

        assert result.summary is not None
        assert result.summary["kind"] in ("graded", "smoke")
        assert "reward_stats" in result.summary
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
        driver = FakeDriver()
        drivers = [(None, driver)]
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "completed",
                "runs": [
                    {
                        "row_index": i,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                    }
                    for i in range(3)
                ],
                "summary": {"kind": "graded", "reward_stats": {"mean": 1.0}},
            },
            completed_runs={(0, 0, None), (1, 0, None), (2, 0, None)},
        )

        orch = _make_orchestrator(drivers=drivers, cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "already_completed"
        assert driver._call_count == 0
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
        driver = FakeDriver()
        drivers = [(None, driver)]
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None)},
        )
        rows = _make_rows(3)

        orch = _make_orchestrator(
            drivers=drivers, cache_backend=cache_backend, rows=rows
        )
        result = await orch.run()

        assert result.status == "completed"
        # Should only run 2 remaining items
        assert driver._call_count == 2
        # Total completed = 1 prior + 2 new
        assert result.total_completed == 3

    @pytest.mark.asyncio
    async def test_resume_includes_prior_runs_in_total(self) -> None:
        """total_completed should include both prior and new runs."""
        driver = FakeDriver()
        drivers = [(None, driver)]
        prior_runs = {(0, 0, None), (1, 0, None)}
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs=prior_runs,
        )

        orch = _make_orchestrator(drivers=drivers, cache_backend=cache_backend)
        result = await orch.run()

        assert result.total_completed == 3
        assert driver._call_count == 1


# ---------------------------------------------------------------------------
# Test: --retry-failed
# ---------------------------------------------------------------------------


class TestOrchestratorRetryFailed:
    @pytest.mark.asyncio
    async def test_retry_failed_requeues_failed_runs_only(self, tmp_path: Path) -> None:
        """Only failed keys should be re-queued when retry_failed=True."""
        driver = FakeDriver()
        drivers = [(None, driver)]
        cache_backend = MockCacheBackend(
            cache_path=tmp_path / "retry_failed_requeue.json",
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": False,
                        "reward": None,
                        "duration_ms": 0,
                        "tokens": 0,
                        "error": "provider error",
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None), (1, 0, None)},
        )
        rows = _make_rows(2)

        orch = _make_orchestrator(
            drivers=drivers,
            cache_backend=cache_backend,
            rows=rows,
            retry_failed=True,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert driver._call_count == 1
        assert result.total_completed == 2
        assert all(run["success"] for run in result.cache_data.get("runs", []))

    @pytest.mark.asyncio
    async def test_retry_failed_cleans_failed_duplicates_for_same_key(
        self, tmp_path: Path
    ) -> None:
        """Failed duplicate records should be pruned when a key already succeeded."""
        driver = FakeDriver()
        drivers = [(None, driver)]
        cache_backend = MockCacheBackend(
            cache_path=tmp_path / "retry_failed_dupes.json",
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": False,
                        "reward": None,
                        "duration_ms": 0,
                        "tokens": 0,
                        "error": "old failure",
                    },
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None)},
        )

        orch = _make_orchestrator(
            drivers=drivers,
            cache_backend=cache_backend,
            rows=_make_rows(1),
            retry_failed=True,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert driver._call_count == 0
        assert len(result.cache_data.get("runs", [])) == 1
        assert result.cache_data["runs"][0]["success"] is True


# ---------------------------------------------------------------------------
# Test: Interrupted via SIGINT
# ---------------------------------------------------------------------------


class TestOrchestratorInterrupted:
    @pytest.mark.asyncio
    async def test_shutdown_event_causes_interrupt(self) -> None:
        """Setting _shutdown_event before run should produce interrupted status."""
        driver = FakeDriver()
        drivers = [(None, driver)]
        orch = _make_orchestrator(drivers=drivers)

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
        assert driver._call_count == 0

    @pytest.mark.asyncio
    async def test_interrupted_records_completed_runs(self) -> None:
        """Runs completed before the interrupt should still be in cache_data."""
        call_count = 0

        class InterruptingDriver(FakeDriver):
            async def run(
                self,
                messages: list[dict[str, Any]],
                label: str | None = None,
                rollout_id: str = "",
            ) -> RolloutOutcome:
                nonlocal call_count
                result = await super().run(messages, label, rollout_id)
                call_count += 1
                # After first call, set shutdown
                if call_count >= 1 and orch._shutdown_event is not None:
                    orch._shutdown_event.set()
                return result

        driver = InterruptingDriver()
        cache_backend = MockCacheBackend()
        rows = _make_rows(5)
        orch = _make_orchestrator(
            drivers=[(None, driver)],
            cache_backend=cache_backend,
            rows=rows,
        )

        result = await orch.run()

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
        driver = FakeDriver()
        orch = _make_orchestrator(
            drivers=[(None, driver)],
            dataset_path=Path("/tmp/data.jsonl"),
            dataset_fingerprint="original_fp",
        )

        # Replace the dataset checker with our mock.
        original_run = orch.run

        async def patched_run() -> OrchestratorResult:
            original_run_all = orch._run_all

            async def run_all_with_mock_checker(
                work_items: list,
                cache_data: dict,
                flush_ctl: Any,
                dataset_checker: Any,
                prior_completed_count: int,
                total_expected: int,
            ) -> tuple:
                mock_checker = MockDatasetChecker(DatasetStatus.MODIFIED)
                return await original_run_all(
                    work_items,
                    cache_data,
                    flush_ctl,
                    mock_checker,
                    prior_completed_count,
                    total_expected,
                )

            orch._run_all = run_all_with_mock_checker  # type: ignore[assignment]
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
        """Systemic error from driver should stop eval with status='systemic_error'."""
        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Auth key expired",
                    systemic_error=True,
                    tokens=0,
                ),
            ]
        )

        orch = _make_orchestrator(drivers=[(None, driver)])
        result = await orch.run()

        assert result.status == "systemic_error"
        assert result.stop_reason is not None
        assert "Auth key expired" in result.stop_reason

    @pytest.mark.asyncio
    async def test_systemic_error_records_failed_run(self) -> None:
        """The failed run should be recorded in cache_data."""
        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Budget exceeded",
                    systemic_error=True,
                    tokens=0,
                ),
            ]
        )

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(drivers=[(None, driver)], cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "systemic_error"
        runs = result.cache_data.get("runs", [])
        assert len(runs) == 1
        assert runs[0]["success"] is False
        assert "Budget exceeded" in runs[0].get("error", "")

    @pytest.mark.asyncio
    async def test_systemic_error_lock_released(self) -> None:
        """Lock should be released even on systemic error."""
        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Connection refused",
                    systemic_error=True,
                    tokens=0,
                ),
            ]
        )

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(drivers=[(None, driver)], cache_backend=cache_backend)
        await orch.run()

        assert cache_backend.lock.released is True


# ---------------------------------------------------------------------------
# Test: Batch mode
# ---------------------------------------------------------------------------


class TestOrchestratorBatchMode:
    @pytest.mark.asyncio
    async def test_batch_mode_completes(self) -> None:
        """Batch mode should complete all work items."""
        driver = FakeDriver()
        rows = _make_rows(3)

        orch = _make_orchestrator(drivers=[(None, driver)], rows=rows, batch_size=3)
        result = await orch.run()

        assert result.status == "completed"
        assert driver._call_count == 3

    @pytest.mark.asyncio
    async def test_batch_mode_records_all_results(self) -> None:
        """All batch results should be recorded in cache_data."""
        driver = FakeDriver()
        rows = _make_rows(4)

        orch = _make_orchestrator(drivers=[(None, driver)], rows=rows, batch_size=2)
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 4
        assert len(result.cache_data.get("runs", [])) == 4

    @pytest.mark.asyncio
    async def test_batch_mode_systemic_error(self) -> None:
        """Systemic error in batch should stop with status='systemic_error'."""
        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Auth failed",
                    systemic_error=True,
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Auth failed",
                    systemic_error=True,
                    tokens=0,
                ),
            ]
        )
        rows = _make_rows(4)

        orch = _make_orchestrator(drivers=[(None, driver)], rows=rows, batch_size=2)
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

        # 3 rows * 2 runs * 1 driver = 6
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

    def test_multiple_drivers(self) -> None:
        """Multiple drivers should multiply work items."""
        d1 = FakeDriver()
        d2 = FakeDriver()
        orch = _make_orchestrator(
            rows=_make_rows(2),
            n_runs=1,
            drivers=[("primary", d1), ("baseline", d2)],
        )
        items = orch._build_work_items(set())

        # 2 rows * 1 run * 2 drivers = 4
        assert len(items) == 4

    def test_model_tag_in_completed_key(self) -> None:
        """Completed runs with model_tag should be correctly matched."""
        d1 = FakeDriver()
        d2 = FakeDriver()
        orch = _make_orchestrator(
            rows=_make_rows(2),
            n_runs=1,
            drivers=[("primary", d1), ("baseline", d2)],
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
        """With log_samples=True and samples with messages, append_sample should be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=True)
        orch._samples_path = Path("/tmp/fake_cache.jsonl")

        cache_data: dict[str, Any] = {"runs": []}
        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": True,
            "reward": 1.0,
            "duration_ms": 100,
            "tokens": 50,
            "model_tag": None,
            "error": None,
            "samples": {
                "s1": RolloutSample(
                    id="s1",
                    messages=[{"role": "assistant", "content": "hello"}],
                    reward=1.0,
                )
            },
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 1
        _, sample = cache_backend.append_sample_calls[0]
        assert sample["row_index"] == 0
        assert sample["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_record_without_log_samples(self) -> None:
        """With log_samples=False, append_sample should NOT be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=False)

        cache_data: dict[str, Any] = {"runs": []}
        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": True,
            "reward": 1.0,
            "duration_ms": 100,
            "tokens": 50,
            "model_tag": None,
            "error": None,
            "samples": {
                "s1": RolloutSample(
                    id="s1",
                    messages=[{"role": "assistant", "content": "hello"}],
                    reward=1.0,
                )
            },
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 0

    def test_record_without_samples(self) -> None:
        """With log_samples=True but no samples, append_sample should NOT be called."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend, log_samples=True)
        orch._samples_path = Path("/tmp/fake_cache.jsonl")

        cache_data: dict[str, Any] = {"runs": []}
        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": False,
            "error": "failed",
            "model_tag": None,
            "samples": {},
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

        assert len(cache_backend.append_sample_calls) == 0

    def test_record_appends_run_dict(self) -> None:
        """Run dict should be appended to cache_data['runs']."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result_dict = {
            "row_index": 5,
            "run_index": 1,
            "success": True,
            "reward": 0.8,
            "duration_ms": 123.0,
            "tokens": 42,
            "model_tag": None,
            "error": None,
            "samples": {},
        }

        orch._record_result(result_dict, cache_data, row_index=5, run_index=1)

        assert len(cache_data["runs"]) == 1
        run_dict = cache_data["runs"][0]
        assert run_dict["row_index"] == 5
        assert run_dict["run_index"] == 1
        assert run_dict["success"] is True
        assert run_dict["reward"] == 0.8
        assert run_dict["duration_ms"] == 123.0
        assert run_dict["tokens"] == 42

    def test_record_includes_model_tag_when_present(self) -> None:
        """model_tag should be included in run dict when not None."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": True,
            "reward": 1.0,
            "model_tag": "primary",
            "samples": {},
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

        assert cache_data["runs"][0]["model_tag"] == "primary"

    def test_record_includes_model_tag_when_none(self) -> None:
        """model_tag should be in run dict even when tag is None."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": True,
            "reward": 1.0,
            "model_tag": None,
            "samples": {},
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

        assert cache_data["runs"][0]["model_tag"] is None

    def test_record_includes_error_when_present(self) -> None:
        """Error message should be in run dict when result has error."""
        orch = _make_orchestrator()
        cache_data: dict[str, Any] = {"runs": []}

        result_dict = {
            "row_index": 0,
            "run_index": 0,
            "success": False,
            "error": "something broke",
            "model_tag": None,
            "samples": {},
        }

        orch._record_result(result_dict, cache_data, row_index=0, run_index=0)

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
        """Lock.release() should be called even when driver raises unexpected error."""

        class ExplodingDriver(RolloutDriver):
            async def run(
                self,
                messages: list[dict[str, Any]],
                label: str | None = None,
                rollout_id: str = "",
            ) -> RolloutOutcome:
                raise RuntimeError("Unexpected crash")

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(
            drivers=[(None, ExplodingDriver())],
            cache_backend=cache_backend,
        )

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
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        rows = _make_rows(3)
        orch = _make_orchestrator(rows=rows, on_progress=progress_cb)
        result = await orch.run()

        assert result.status == "completed"
        assert len(progress_calls) == 3

    @pytest.mark.asyncio
    async def test_on_progress_total_matches_expected(self) -> None:
        """Total in progress callback should be rows * n_runs * drivers."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        rows = _make_rows(2)
        orch = _make_orchestrator(rows=rows, n_runs=2, on_progress=progress_cb)
        await orch.run()

        # 2 rows * 2 runs * 1 driver = 4
        assert len(progress_calls) == 4
        for _, total, _ in progress_calls:
            assert total == 4

    @pytest.mark.asyncio
    async def test_on_progress_with_multiple_drivers(self) -> None:
        """Total should account for multiple drivers."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        d1 = FakeDriver()
        d2 = FakeDriver()
        rows = _make_rows(2)
        orch = _make_orchestrator(
            rows=rows,
            n_runs=1,
            drivers=[("primary", d1), ("baseline", d2)],
            on_progress=progress_cb,
        )
        await orch.run()

        # 2 rows * 1 run * 2 drivers = 4
        assert len(progress_calls) == 4
        for _, total, _ in progress_calls:
            assert total == 4

    @pytest.mark.asyncio
    async def test_on_progress_current_increments(self) -> None:
        """current values should be monotonically increasing."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        orch = _make_orchestrator(rows=_make_rows(3), on_progress=progress_cb)
        await orch.run()

        currents = [c for c, _, _ in progress_calls]
        assert currents == sorted(currents)
        assert currents[-1] == 3

    @pytest.mark.asyncio
    async def test_on_progress_resume_includes_prior_runs_sequential(self) -> None:
        """Resume mode progress should include prior completed runs."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None)},
        )

        orch = _make_orchestrator(
            rows=_make_rows(3),
            cache_backend=cache_backend,
            on_progress=progress_cb,
        )
        await orch.run()

        currents = [c for c, _, _ in progress_calls]
        assert currents == [2, 3]
        for _, total, _ in progress_calls:
            assert total == 3

    @pytest.mark.asyncio
    async def test_on_progress_resume_includes_prior_runs_batched(self) -> None:
        """Batched resume progress should include prior completed runs."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None), (1, 0, None)},
        )

        orch = _make_orchestrator(
            rows=_make_rows(4),
            batch_size=2,
            cache_backend=cache_backend,
            on_progress=progress_cb,
        )
        await orch.run()

        currents = [c for c, _, _ in progress_calls]
        assert currents == [3, 4]
        for _, total, _ in progress_calls:
            assert total == 4

    @pytest.mark.asyncio
    async def test_on_progress_called_on_systemic_error(self) -> None:
        """Progress should be reported even for the failing run."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="Error",
                    systemic_error=True,
                    tokens=0,
                ),
            ]
        )

        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=_make_rows(3),
            on_progress=progress_cb,
        )
        result = await orch.run()

        assert result.status == "systemic_error"
        assert len(progress_calls) == 1
        _, _, last_result = progress_calls[0]
        assert last_result["success"] is False


# ---------------------------------------------------------------------------
# Test: Flush controller
# ---------------------------------------------------------------------------


class TestFlushController:
    @pytest.mark.asyncio
    async def test_force_flush_called_on_normal_completion(self) -> None:
        """force_flush() should be called in the finally block."""
        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(cache_backend=cache_backend)
        result = await orch.run()

        assert result.status == "completed"
        assert cache_backend.lock.released is True

    @pytest.mark.asyncio
    async def test_force_flush_called_on_error(self) -> None:
        """force_flush() should be called even when an error occurs."""

        class ExplodingDriver(RolloutDriver):
            async def run(
                self,
                messages: list[dict[str, Any]],
                label: str | None = None,
                rollout_id: str = "",
            ) -> RolloutOutcome:
                raise RuntimeError("Boom")

        cache_backend = MockCacheBackend()
        orch = _make_orchestrator(
            drivers=[(None, ExplodingDriver())],
            cache_backend=cache_backend,
        )

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
            summary={"kind": "smoke", "reward_stats": None},
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
        driver = FakeDriver()
        orch = _make_orchestrator(drivers=[(None, driver)], rows=[])
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 0
        assert result.total_expected == 0
        assert driver._call_count == 0

    @pytest.mark.asyncio
    async def test_all_runs_completed_but_status_not_completed(self) -> None:
        """When all runs exist in cache but status is in_progress, should complete."""
        driver = FakeDriver()
        rows = _make_rows(2)
        cache_backend = MockCacheBackend(
            cache_data={
                "status": "in_progress",
                "runs": [
                    {
                        "row_index": 0,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                    {
                        "row_index": 1,
                        "run_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "duration_ms": 0,
                        "tokens": 0,
                    },
                ],
                "summary": None,
            },
            completed_runs={(0, 0, None), (1, 0, None)},
        )

        orch = _make_orchestrator(
            drivers=[(None, driver)], cache_backend=cache_backend, rows=rows
        )
        result = await orch.run()

        assert result.status == "completed"
        assert driver._call_count == 0
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
        driver = FakeDriver()
        orch = _make_orchestrator(
            drivers=[(None, driver)], rows=_make_rows(2), n_runs=3
        )
        result = await orch.run()

        assert result.total_expected == 6  # 2 rows * 3 runs
        assert result.total_completed == 6
        assert driver._call_count == 6

    @pytest.mark.asyncio
    async def test_batch_progress_callback(self) -> None:
        """Progress callback should work in batch mode too."""
        progress_calls: list[tuple[int, int, dict]] = []

        def progress_cb(current: int, total: int, result: dict) -> None:
            progress_calls.append((current, total, result))

        driver = FakeDriver()
        rows = _make_rows(3)
        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=rows,
            batch_size=2,
            on_progress=progress_cb,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert len(progress_calls) == 3


# ---------------------------------------------------------------------------
# Test: V2-specific features — baseline comparison
# ---------------------------------------------------------------------------


class TestOrchestratorBaselineComparison:
    @pytest.mark.asyncio
    async def test_orchestrator_baseline_comparison(self) -> None:
        """Two drivers with model tags should both be called for every row."""
        d_primary = FakeDriver()
        d_baseline = FakeDriver()
        rows = _make_rows(3)

        orch = _make_orchestrator(
            drivers=[("primary", d_primary), ("baseline", d_baseline)],
            rows=rows,
            n_runs=1,
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_expected == 6  # 3 rows * 1 run * 2 drivers
        assert result.total_completed == 6
        assert d_primary._call_count == 3
        assert d_baseline._call_count == 3

        # Verify model_tags are recorded in runs
        runs = result.cache_data.get("runs", [])
        tags = {r["model_tag"] for r in runs}
        assert tags == {"primary", "baseline"}


# ---------------------------------------------------------------------------
# Test: V2-specific features — concurrency cap
# ---------------------------------------------------------------------------


class TestOrchestratorConcurrencyCap:
    @pytest.mark.asyncio
    async def test_orchestrator_concurrency_cap(self) -> None:
        """driver.max_concurrency should be respected as upper bound on batch size."""
        driver = FakeDriver(max_conc=2)
        rows = _make_rows(6)

        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=rows,
            batch_size=10,  # Larger than max_concurrency
        )
        result = await orch.run()

        assert result.status == "completed"
        assert result.total_completed == 6
        assert driver._call_count == 6
        # Peak concurrency should not exceed max_concurrency
        assert driver._peak_concurrent <= 2


# ---------------------------------------------------------------------------
# Test: V2-specific features — zero-token fail-fast
# ---------------------------------------------------------------------------


class TestOrchestratorZeroTokenFailFast:
    @pytest.mark.asyncio
    async def test_zero_token_failfast_sequential(self) -> None:
        """2 consecutive zero-token failures should stop the eval."""
        driver = FakeDriver(
            outcomes=[
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                # Third should never be reached
                RolloutOutcome(
                    status=RolloutStatus.SUCCESS,
                    samples={"s1": RolloutSample(id="s1", messages=[], reward=1.0)},
                    tokens=50,
                ),
            ]
        )

        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=_make_rows(5),
            batch_size=1,
        )
        result = await orch.run()

        assert result.status == "systemic_error"
        assert result.stop_reason is not None
        assert "zero-token" in result.stop_reason.lower()
        assert driver._call_count == 2

    @pytest.mark.asyncio
    async def test_zero_token_failfast_concurrent(self) -> None:
        """2 consecutive all-zero-token batches should stop the eval."""
        driver = FakeDriver(
            outcomes=[
                # Batch 1 (2 items): all zero-token
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                # Batch 2 (2 items): all zero-token
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                # Batch 3: should never be reached
                RolloutOutcome(
                    status=RolloutStatus.SUCCESS,
                    samples={"s1": RolloutSample(id="s1", messages=[], reward=1.0)},
                    tokens=50,
                ),
            ]
        )

        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=_make_rows(6),
            batch_size=2,
        )
        result = await orch.run()

        assert result.status == "systemic_error"
        assert result.stop_reason is not None
        assert "zero-token" in result.stop_reason.lower()
        assert driver._call_count == 4  # 2 batches of 2

    @pytest.mark.asyncio
    async def test_zero_token_mixed_batch_no_stop(self) -> None:
        """A mixed batch (some success, some zero-token) should reset the counter."""
        driver = FakeDriver(
            outcomes=[
                # Batch 1: all zero-token failures
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                # Batch 2: mixed — one success resets the counter
                RolloutOutcome(
                    status=RolloutStatus.SUCCESS,
                    samples={"s1": RolloutSample(id="s1", messages=[], reward=1.0)},
                    tokens=50,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                # Batch 3: all zero-token — only 1 consecutive, not 2
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
                RolloutOutcome(
                    status=RolloutStatus.FAILURE,
                    error="No tokens",
                    tokens=0,
                ),
            ]
        )

        orch = _make_orchestrator(
            drivers=[(None, driver)],
            rows=_make_rows(6),
            batch_size=2,
        )
        result = await orch.run()

        # After batch 2 resets the counter, batch 3 is only 1 consecutive zero batch.
        # The eval should complete (not stopped for zero-token).
        assert result.status == "completed"
        assert driver._call_count == 6
