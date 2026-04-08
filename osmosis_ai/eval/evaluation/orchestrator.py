"""Evaluation orchestrator sitting between CLI and RolloutDriver.

Manages cache lifecycle, signal handling, dataset integrity checking,
periodic flushing, and progress reporting while delegating actual
agent execution to RolloutDriver instances.
"""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.eval.common.dataset import DatasetRow
from osmosis_ai.eval.evaluation.cache import (
    BuildSummaryResult,
    CacheBackend,
    CacheConfig,
    CacheFlushController,
    DatasetIntegrityChecker,
    DatasetStatus,
    build_summary,
)
from osmosis_ai.rollout_v2.driver import RolloutDriver
from osmosis_ai.rollout_v2.types import RolloutSample, RolloutStatus


def _extract_mean_reward(samples: dict[str, RolloutSample]) -> float | None:
    """Compute the mean reward from a samples dict, ignoring None rewards."""
    rewards = [s.reward for s in samples.values() if s.reward is not None]
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


@dataclass
class OrchestratorResult:
    """Result returned by :meth:`EvalOrchestrator.run`.

    Attributes:
        status: One of ``"completed"``, ``"interrupted"``, ``"dataset_modified"``,
            ``"already_completed"``, or ``"systemic_error"``.
        cache_path: Path to the cache JSON file.
        samples_path: Path to the JSONL samples file, or ``None``.
        summary: Aggregated summary dict, or ``None`` if not completed.
        total_completed: Number of runs that were completed (including prior).
        total_expected: Total number of runs expected.
        cache_data: The full cache data dict.
        stop_reason: Human-readable reason the eval stopped early, or ``None``.
        dataset_fingerprint_warning: Warning message if the dataset file has changed
            since the cached evaluation completed, or ``None``.
    """

    status: str
    cache_path: Path
    samples_path: Path | None
    summary: BuildSummaryResult | None
    total_completed: int
    total_expected: int
    cache_data: dict[str, Any]
    stop_reason: str | None = None
    dataset_fingerprint_warning: str | None = None


class EvalOrchestrator:
    """Orchestrates an evaluation run with caching, resumption, and signal handling.

    Sits between the CLI layer and :class:`RolloutDriver`, managing:
    - Cache acquisition and lock management
    - Work item construction (with resume support)
    - Signal-based graceful interruption
    - Periodic dataset integrity checks
    - Periodic cache flushing
    - Final summary computation
    """

    def __init__(
        self,
        drivers: list[tuple[str | None, RolloutDriver]],
        cache_backend: CacheBackend,
        cache_config: CacheConfig,
        rows: list[DatasetRow],
        n_runs: int = 1,
        pass_threshold: float = 1.0,
        batch_size: int = 1,
        log_samples: bool = False,
        fresh: bool = False,
        retry_failed: bool = False,
        dataset_path: Path | None = None,
        dataset_fingerprint: str | None = None,
        start_index: int = 0,
        on_progress: Callable[[int, int, dict], None] | None = None,
    ):
        self.drivers = drivers
        self.model_tags: list[str | None] = [tag for tag, _ in self.drivers]
        self.cache_backend = cache_backend
        self.cache_config = cache_config
        self.rows = rows
        self.n_runs = n_runs
        self.pass_threshold = pass_threshold
        self.batch_size = batch_size
        self.log_samples = log_samples
        self.fresh = fresh
        self.retry_failed = retry_failed
        self.dataset_path = dataset_path
        self.dataset_fingerprint = dataset_fingerprint
        self.start_index = start_index
        self.on_progress = on_progress

        # Runtime state (set during run())
        self._samples_path: Path | None = None

        # Signal handling state
        self._shutdown_event: asyncio.Event | None = None
        self._using_loop_signals: bool = False
        self._original_sigint_handler: Any = None
        self._original_sigterm_handler: Any = None

    async def run(self) -> OrchestratorResult:
        """Execute the evaluation orchestration.

        Returns:
            An :class:`OrchestratorResult` describing the outcome.
        """
        cfg = self.cache_config
        total_expected = len(self.rows) * self.n_runs * len(self.drivers)

        lock = self.cache_backend.acquire_lock(cfg.task_id, cfg.model, cfg.dataset_path)
        try:
            cache_path, cache_data, completed_runs = (
                self.cache_backend.lookup_or_create(
                    task_id=cfg.task_id,
                    config_hash=cfg.config_hash,
                    fresh=self.fresh,
                    config=cfg.config,
                    total_rows=cfg.total_rows,
                    model=cfg.model,
                    dataset_path=cfg.dataset_path,
                    dataset_fingerprint=self.dataset_fingerprint,
                )
            )

            samples_path = (
                cache_path.with_suffix(".jsonl") if self.log_samples else None
            )
            self._samples_path = samples_path

            # Handle --retry-failed: keep only successful runs in completed set.
            if self.retry_failed and completed_runs:
                all_runs = cache_data.get("runs", [])
                successful_runs = [r for r in all_runs if r.get("success", True)]
                successful_keys = {
                    (r["row_index"], r["run_index"], r.get("model_tag"))
                    for r in successful_runs
                }
                completed_runs = successful_keys

                # Write back a cleaned cache only if failed records existed.
                failed_count = len(all_runs) - len(successful_runs)
                if failed_count > 0:
                    cache_data["runs"] = successful_runs
                    cache_data["status"] = "in_progress"
                    self.cache_backend.write_cache(cache_path, cache_data)

            # Case C: already completed
            if cache_data.get("status") == "completed":
                dataset_fingerprint_warning: str | None = None
                if self.dataset_fingerprint is not None:
                    cached_fp = cache_data.get("config", {}).get("dataset_fingerprint")
                    if cached_fp and self.dataset_fingerprint != cached_fp:
                        dataset_fingerprint_warning = (
                            f"Warning: Dataset file has changed since this eval completed.\n"
                            f"  Cached: {cached_fp[:16]}... | Current: {self.dataset_fingerprint[:16]}...\n"
                            f"  Results below are from the original dataset."
                        )
                return OrchestratorResult(
                    status="already_completed",
                    cache_path=cache_path,
                    samples_path=samples_path,
                    summary=cache_data.get("summary"),
                    total_completed=len(completed_runs),
                    total_expected=total_expected,
                    cache_data=cache_data,
                    dataset_fingerprint_warning=dataset_fingerprint_warning,
                )

            # Build work items, skipping already-completed runs
            work_items = self._build_work_items(completed_runs)

            # Strip prior runs from in-memory cache_data so CacheFlushController
            # can properly merge old (from disk) + new (in-memory) without duplication.
            # The flush controller reads old runs from disk and prepends them,
            # so cache_data["runs"] must only contain newly-appended runs.
            # Save original runs first as fallback for the no-work-items path.
            prior_runs_snapshot = list(cache_data.get("runs", []))
            if completed_runs:
                cache_data["runs"] = []

            if not work_items:
                # All runs already completed but status wasn't "completed".
                # Re-read runs from disk since we may have cleared cache_data["runs"]
                # above for the flush controller.
                try:
                    disk_data = json.loads(cache_path.read_text())
                    all_runs = disk_data.get("runs", [])
                except (ValueError, OSError):
                    all_runs = prior_runs_snapshot
                summary = build_summary(
                    all_runs,
                    self.pass_threshold,
                    self.n_runs,
                )
                cache_data["runs"] = all_runs
                cache_data["status"] = "completed"
                cache_data["summary"] = summary
                self.cache_backend.write_cache(cache_path, cache_data)
                return OrchestratorResult(
                    status="completed",
                    cache_path=cache_path,
                    samples_path=samples_path,
                    summary=summary,
                    total_completed=len(completed_runs),
                    total_expected=total_expected,
                    cache_data=cache_data,
                )

            # Install signal handlers for graceful shutdown
            self._shutdown_event = asyncio.Event()
            self._install_signal_handlers()

            # Setup cache flush controller and dataset integrity checker
            flush_ctl = CacheFlushController(
                cache_path=cache_path,
                cache_data=cache_data,
                prior_runs_count=len(completed_runs),
            )
            prior_completed_count = len(completed_runs)

            dataset_checker: DatasetIntegrityChecker | None = None
            if self.dataset_path is not None and self.dataset_fingerprint is not None:
                dataset_checker = DatasetIntegrityChecker(
                    dataset_path=self.dataset_path,
                    expected_fingerprint=self.dataset_fingerprint,
                )

            try:
                (
                    current,
                    interrupted,
                    dataset_modified,
                    stop_reason,
                ) = await self._run_all(
                    work_items,
                    cache_data,
                    flush_ctl,
                    dataset_checker,
                    prior_completed_count,
                    total_expected,
                )
            finally:
                flush_ctl.force_flush()

            # Determine final status
            total_completed = len(completed_runs) + current

            if dataset_modified:
                status = "dataset_modified"
            elif interrupted:
                status = "interrupted"
            elif stop_reason is not None:
                status = "systemic_error"
            else:
                status = "completed"

            summary: BuildSummaryResult | None = None
            if status == "completed":
                # After force_flush, disk has the full merged runs list.
                # Re-read from disk to get all runs (old + new) for summary.
                # Fall back to in-memory runs (which include at least the new
                # runs from this session) to avoid data loss if disk read fails.
                try:
                    disk_data = json.loads(cache_path.read_text())
                    all_runs = disk_data.get("runs", [])
                except (ValueError, OSError):
                    all_runs = prior_runs_snapshot + cache_data.get("runs", [])
                summary = build_summary(
                    all_runs,
                    self.pass_threshold,
                    self.n_runs,
                )
                cache_data["runs"] = all_runs
                cache_data["status"] = "completed"
                cache_data["summary"] = summary
                self.cache_backend.write_cache(cache_path, cache_data)

            return OrchestratorResult(
                status=status,
                cache_path=cache_path,
                samples_path=samples_path,
                summary=summary,
                total_completed=total_completed,
                total_expected=total_expected,
                cache_data=cache_data,
                stop_reason=stop_reason,
            )
        finally:
            self._remove_signal_handlers()
            lock.release()

    def _build_work_items(
        self, completed_runs: set[tuple[int, int, str | None]]
    ) -> list[tuple[DatasetRow, int, int, str | None, RolloutDriver]]:
        """Build list of work items, skipping those already completed.

        Returns:
            List of ``(row, row_index, run_index, model_tag, driver)`` tuples.
        """
        items: list[tuple[DatasetRow, int, int, str | None, RolloutDriver]] = []
        for i, row in enumerate(self.rows):
            row_index = self.start_index + i
            for run_idx in range(self.n_runs):
                for tag, driver in self.drivers:
                    if (row_index, run_idx, tag) not in completed_runs:
                        items.append((row, row_index, run_idx, tag, driver))
        return items

    async def _execute_one(
        self,
        row: DatasetRow,
        row_index: int,
        run_index: int,
        model_tag: str | None,
        driver: RolloutDriver,
    ) -> dict[str, Any]:
        """Execute a single rollout and return a result dict."""
        from osmosis_ai.eval.common.dataset import dataset_row_to_prompt

        rollout_id = f"eval-{row_index}-run-{run_index}"
        if model_tag:
            rollout_id += f"-{model_tag}"

        outcome = await driver.run(
            messages=dataset_row_to_prompt(row),
            label=row.get("ground_truth"),
            rollout_id=rollout_id,
        )

        reward = _extract_mean_reward(outcome.samples) if outcome.samples else None

        return {
            "row_index": row_index,
            "run_index": run_index,
            "success": outcome.status == RolloutStatus.SUCCESS,
            "reward": reward,
            "duration_ms": outcome.duration_ms,
            "tokens": outcome.tokens,
            "model_tag": model_tag,
            "error": outcome.error,
            "systemic_error": outcome.systemic_error,
            "samples": outcome.samples,
        }

    async def _run_all(
        self,
        work_items: list[tuple[DatasetRow, int, int, str | None, RolloutDriver]],
        cache_data: dict[str, Any],
        flush_ctl: CacheFlushController,
        dataset_checker: DatasetIntegrityChecker | None,
        prior_completed_count: int,
        total_expected: int,
    ) -> tuple[int, bool, bool, str | None]:
        """Execute work items using driver.run() with semaphore-based concurrency.

        Supports both sequential (batch_size<=1) and concurrent execution.
        Includes zero-token fail-fast heuristic: stops after 2 consecutive
        zero-token failures (sequential) or 2 consecutive all-zero-token
        batches (concurrent).

        Returns:
            ``(current, interrupted, dataset_modified, stop_reason)``
        """
        current = 0
        interrupted = False
        dataset_modified = False
        stop_reason: str | None = None

        # Compute effective batch size, respecting driver max_concurrency.
        # Use the first driver's max_concurrency as representative.
        _, first_driver = self.drivers[0]
        driver_max = first_driver.max_concurrency
        if driver_max > 0:
            effective_batch = min(self.batch_size, driver_max)
        else:
            effective_batch = self.batch_size

        if effective_batch <= 1:
            # ---------- Sequential path ----------
            consecutive_zero_token = 0
            for item in work_items:
                if self._shutdown_event is not None and self._shutdown_event.is_set():
                    interrupted = True
                    break

                if dataset_checker is not None:
                    ds_status = dataset_checker.maybe_check()
                    if ds_status != DatasetStatus.VALID:
                        dataset_modified = True
                        stop_reason = f"Dataset {ds_status.value}"
                        break

                row, row_index, run_idx, tag, driver = item
                result_dict = await self._execute_one(
                    row, row_index, run_idx, tag, driver
                )

                self._record_result(result_dict, cache_data, row_index, run_idx)
                current += 1
                flush_ctl.maybe_flush(runs_completed=1)

                if self.on_progress is not None:
                    self.on_progress(
                        prior_completed_count + current, total_expected, result_dict
                    )

                # Systemic error → stop immediately
                if result_dict.get("systemic_error"):
                    stop_reason = result_dict.get("error", "Systemic error")
                    break

                # Zero-token fail-fast heuristic
                if result_dict.get("tokens", 0) == 0 and not result_dict["success"]:
                    consecutive_zero_token += 1
                    if consecutive_zero_token >= 2:
                        stop_reason = (
                            "Stopped after 2 consecutive zero-token failures. "
                            "Check your LLM endpoint configuration."
                        )
                        break
                else:
                    consecutive_zero_token = 0
        else:
            # ---------- Concurrent path ----------
            semaphore = asyncio.Semaphore(effective_batch)
            cursor = 0
            consecutive_zero_batches = 0

            while cursor < len(work_items):
                if self._shutdown_event is not None and self._shutdown_event.is_set():
                    interrupted = True
                    break

                if dataset_checker is not None:
                    ds_status = dataset_checker.maybe_check()
                    if ds_status != DatasetStatus.VALID:
                        dataset_modified = True
                        stop_reason = f"Dataset {ds_status.value}"
                        break

                batch = work_items[cursor : cursor + effective_batch]

                batch_results: list[dict[str, Any] | None] = [None] * len(batch)

                async def _run_with_semaphore(
                    idx: int,
                    row: DatasetRow,
                    row_index: int,
                    run_idx: int,
                    tag: str | None,
                    drv: RolloutDriver,
                ) -> None:
                    async with semaphore:
                        result = await self._execute_one(
                            row, row_index, run_idx, tag, drv
                        )
                        batch_results[idx] = result  # noqa: B023

                tasks = [
                    asyncio.create_task(
                        _run_with_semaphore(idx, row, row_index, run_idx, tag, drv)
                    )
                    for idx, (row, row_index, run_idx, tag, drv) in enumerate(batch)
                ]

                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

                # Process batch results
                batch_has_systemic = False
                all_zero_tokens = True
                completed_in_batch = 0

                for result_dict in batch_results:
                    if result_dict is None:
                        continue
                    row_index = result_dict["row_index"]
                    run_idx = result_dict["run_index"]
                    self._record_result(result_dict, cache_data, row_index, run_idx)
                    current += 1
                    completed_in_batch += 1

                    if self.on_progress is not None:
                        self.on_progress(
                            prior_completed_count + current,
                            total_expected,
                            result_dict,
                        )

                    if result_dict.get("systemic_error"):
                        batch_has_systemic = True
                        stop_reason = result_dict.get("error", "Systemic error")

                    # Track zero-token status for the batch
                    if result_dict.get("tokens", 0) > 0 or result_dict["success"]:
                        all_zero_tokens = False

                flush_ctl.maybe_flush(runs_completed=completed_in_batch)
                cursor += len(batch)

                if batch_has_systemic:
                    break

                # Zero-token fail-fast at batch level
                if all_zero_tokens and completed_in_batch > 0:
                    consecutive_zero_batches += 1
                    if consecutive_zero_batches >= 2:
                        stop_reason = (
                            "Stopped after 2 consecutive zero-token batches. "
                            "Check your LLM endpoint configuration."
                        )
                        break
                else:
                    consecutive_zero_batches = 0

        return current, interrupted, dataset_modified, stop_reason

    def _record_result(
        self,
        result_dict: dict[str, Any],
        cache_data: dict[str, Any],
        row_index: int,
        run_index: int,
    ) -> None:
        """Record a single run result into cache data and optionally log samples."""
        if self.log_samples and self._samples_path is not None:
            samples = result_dict.get("samples", {})
            if samples:
                messages: list[Any] = []
                for s in samples.values():
                    msgs = s.messages if hasattr(s, "messages") else []
                    messages.extend(msgs)
                if messages:
                    sample = {
                        "row_index": row_index,
                        "run_index": run_index,
                        "model_tag": result_dict.get("model_tag"),
                        "messages": messages,
                    }
                    self.cache_backend.append_sample(self._samples_path, sample)

        run_dict: dict[str, Any] = {
            "row_index": row_index,
            "run_index": run_index,
            "success": result_dict["success"],
            "reward": result_dict.get("reward"),
            "duration_ms": result_dict.get("duration_ms", 0),
            "tokens": result_dict.get("tokens", 0),
            "model_tag": result_dict.get("model_tag"),
            "error": result_dict.get("error"),
        }

        cache_data["runs"].append(run_dict)

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that set the shutdown event."""
        if self._shutdown_event is None:
            return

        shutdown_event = self._shutdown_event

        def _signal_handler(*_: Any) -> None:
            shutdown_event.set()

        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _signal_handler)
            loop.add_signal_handler(signal.SIGTERM, _signal_handler)
            self._using_loop_signals = True
        except NotImplementedError:
            # Windows: fall back to signal.signal()
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            self._using_loop_signals = False

    def _remove_signal_handlers(self) -> None:
        """Remove previously installed signal handlers."""
        if self._shutdown_event is None:
            return

        with suppress(Exception):
            if self._using_loop_signals:
                loop = asyncio.get_running_loop()
                with suppress(Exception):
                    loop.remove_signal_handler(signal.SIGINT)
                with suppress(Exception):
                    loop.remove_signal_handler(signal.SIGTERM)
            else:
                if self._original_sigint_handler is not None:
                    signal.signal(signal.SIGINT, self._original_sigint_handler)
                if self._original_sigterm_handler is not None:
                    signal.signal(signal.SIGTERM, self._original_sigterm_handler)


__all__ = [
    "EvalOrchestrator",
    "OrchestratorResult",
]
