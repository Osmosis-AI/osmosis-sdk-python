"""Evaluation runner for executing agent loops with eval functions.

Runs the agent against each dataset row (optionally multiple times for pass@n),
applies eval functions to each result, and aggregates statistics.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.eval.common.dataset import DatasetRow
from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
from osmosis_ai.rollout.eval.common.runner import LocalRolloutRunner
from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper

logger: logging.Logger = logging.getLogger(__name__)


def _extract_systemic_error_metrics(
    e: SystemicProviderError,
    fallback_started_at: float,
) -> tuple[float, int]:
    """Extract (duration_ms, tokens) from a systemic error with sane fallbacks."""
    duration_ms = getattr(e, "duration_ms", None)
    if duration_ms is None:
        duration_ms = (time.monotonic() - fallback_started_at) * 1000

    tokens = getattr(e, "tokens", None)
    if tokens is None:
        tokens = 0

    return float(duration_ms), int(tokens)


@dataclass
class EvalRunResult:
    """Result from a single agent run + eval scoring.

    Attributes:
        run_index: Which run this is (0-indexed within a row).
        success: Whether the agent completed successfully.
        scores: Eval function name -> score mapping.
        duration_ms: Execution time in milliseconds.
        tokens: Total tokens used.
        error: Error message if agent execution failed.
        model_tag: "primary", "baseline", or None for single-model mode.
        messages: Final conversation messages from the agent run, or None on failure.
        row_index: The dataset row index this result belongs to.
    """

    run_index: int
    success: bool
    scores: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    tokens: int = 0
    error: str | None = None
    model_tag: str | None = None
    messages: list[dict[str, Any]] | None = None
    row_index: int = 0


@dataclass
class EvalRowResult:
    """Results from all runs of a single dataset row.

    Attributes:
        row_index: Index of the row in the dataset.
        runs: List of run results.
    """

    row_index: int
    runs: list[EvalRunResult] = field(default_factory=list)


@dataclass
class EvalEvalSummary:
    """Summary statistics for a single eval function across all runs.

    Attributes:
        mean: Mean score.
        std: Standard deviation of scores.
        min: Minimum score.
        max: Maximum score.
        pass_at_k: Dict of k -> pass@k value. Only populated when n > 1.
    """

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)


@dataclass
class EvalModelSummary:
    """Per-model summary (comparison mode only).

    Attributes:
        model: Model identifier string.
        model_tag: "primary" or "baseline".
        eval_summaries: Per-eval-function summary statistics for this model.
        total_runs: Number of runs for this model.
        total_tokens: Total tokens consumed by this model.
        total_duration_ms: Total wall time for this model's runs.
    """

    model: str
    model_tag: str
    eval_summaries: dict[str, EvalEvalSummary]
    total_runs: int
    total_tokens: int
    total_duration_ms: float


@dataclass
class EvalResult:
    """Full evaluation result with per-row data and aggregated summaries.

    Attributes:
        rows: Per-row results.
        eval_summaries: Per-eval-function summary statistics.
        total_rows: Number of dataset rows.
        total_runs: Total number of runs (rows * n_runs).
        total_tokens: Total tokens consumed.
        total_duration_ms: Total wall time.
        n_runs: Number of runs per row.
        pass_threshold: Score threshold for pass@k.
        model_summaries: Per-model summaries (comparison mode only).
    """

    rows: list[EvalRowResult]
    eval_summaries: dict[str, EvalEvalSummary]
    total_rows: int
    total_runs: int
    total_tokens: int
    total_duration_ms: float
    n_runs: int
    pass_threshold: float
    stopped_early: bool = False
    stop_reason: str | None = None
    model_summaries: list[EvalModelSummary] | None = None


class EvalRunner:
    """Orchestrates evaluation execution: agent runs + eval function scoring."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: ExternalLLMClient,
        eval_fns: list[EvalFnWrapper],
        debug: bool = False,
        debug_dir: str | None = None,
        llm_client_factory: Callable[[], Any] | None = None,
        baseline_llm_client: ExternalLLMClient | None = None,
        baseline_llm_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.llm_client = llm_client
        self.eval_fns = eval_fns
        self.debug = debug
        self.debug_dir = debug_dir
        self._llm_client_factory = (
            llm_client_factory or self._default_llm_client_factory
        )
        self._rollout_runner = LocalRolloutRunner(
            agent_loop=agent_loop,
            llm_client=llm_client,
            debug=debug,
            debug_dir=debug_dir,
            rollout_id_prefix="eval",
            request_metadata={"execution_mode": "eval"},
        )

        # Baseline model support (comparison mode)
        self._baseline_llm_client = baseline_llm_client
        self._baseline_llm_client_factory = (
            baseline_llm_client_factory or self._default_baseline_llm_client_factory
        )
        self._baseline_rollout_runner: LocalRolloutRunner | None = None
        if baseline_llm_client is not None:
            self._baseline_rollout_runner = LocalRolloutRunner(
                agent_loop=agent_loop,
                llm_client=baseline_llm_client,
                debug=debug,
                debug_dir=debug_dir,
                rollout_id_prefix="eval-baseline",
                request_metadata={"execution_mode": "eval", "model_tag": "baseline"},
            )

    def _default_llm_client_factory(self) -> ExternalLLMClient:
        """Create a new ExternalLLMClient from the original client's config."""
        return ExternalLLMClient(
            model=self.llm_client.display_name,
            api_key=self.llm_client.api_key,
            api_base=self.llm_client.api_base,
        )

    def _default_baseline_llm_client_factory(self) -> ExternalLLMClient:
        """Create a new ExternalLLMClient from the baseline client's config."""
        if self._baseline_llm_client is None:
            raise RuntimeError("No baseline LLM client configured")
        return ExternalLLMClient(
            model=self._baseline_llm_client.display_name,
            api_key=self._baseline_llm_client.api_key,
            api_base=self._baseline_llm_client.api_base,
        )

    def _create_rollout_runner(self) -> LocalRolloutRunner:
        """Create a new LocalRolloutRunner with a fresh LLM client.

        Each runner gets its own client instance so concurrent runs
        don't share mutable metrics/tools state.
        """
        return LocalRolloutRunner(
            agent_loop=self.agent_loop,
            llm_client=self._llm_client_factory(),
            debug=self.debug,
            debug_dir=self.debug_dir,
            rollout_id_prefix="eval",
            request_metadata={"execution_mode": "eval"},
        )

    def _create_baseline_rollout_runner(self) -> LocalRolloutRunner:
        """Create a new LocalRolloutRunner with a fresh baseline LLM client."""
        return LocalRolloutRunner(
            agent_loop=self.agent_loop,
            llm_client=self._baseline_llm_client_factory(),
            debug=self.debug,
            debug_dir=self.debug_dir,
            rollout_id_prefix="eval-baseline",
            request_metadata={"execution_mode": "eval", "model_tag": "baseline"},
        )

    @property
    def has_baseline(self) -> bool:
        """Whether a baseline model is configured for comparison."""
        return self._baseline_llm_client is not None

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        run_index: int,
        max_turns: int = 10,
        completion_params: dict[str, Any] | None = None,
        runner: LocalRolloutRunner | None = None,
        model_tag: str | None = None,
    ) -> EvalRunResult:
        """Run agent once on a row and apply all eval functions.

        Args:
            row: Dataset row.
            row_index: Index of the row in the dataset.
            run_index: Which run this is (for pass@n).
            max_turns: Max agent turns.
            completion_params: LLM sampling parameters.
            runner: Optional runner instance for concurrent execution.
            model_tag: "primary", "baseline", or None for single-model mode.

        Returns:
            EvalRunResult with scores from all eval functions.
        """
        # Run the agent — select runner based on model_tag
        if runner is not None:
            rollout_runner = runner
        elif model_tag == "baseline" and self._baseline_rollout_runner is not None:
            rollout_runner = self._baseline_rollout_runner
        else:
            rollout_runner = self._rollout_runner
        # Include model_tag in rollout ID so primary and baseline runs don't
        # collide (debug traces are written as {rollout_id}.jsonl).
        if model_tag:
            rollout_id = f"eval-{row_index}-run-{run_index}-{model_tag}"
        else:
            rollout_id = f"eval-{row_index}-run-{run_index}"
        run_start = time.monotonic()
        try:
            test_result = await rollout_runner.run_single(
                row=row,
                row_index=row_index,
                max_turns=max_turns,
                completion_params=completion_params,
                rollout_id=rollout_id,
                request_metadata={"run_index": run_index},
            )
        except SystemicProviderError as e:
            duration_ms, tokens = _extract_systemic_error_metrics(
                e,
                fallback_started_at=run_start,
            )
            e.duration_ms = duration_ms
            e.tokens = tokens
            raise

        if not test_result.success or test_result.result is None:
            return EvalRunResult(
                run_index=run_index,
                success=False,
                duration_ms=test_result.duration_ms,
                tokens=test_result.token_usage.get("total_tokens", 0),
                error=test_result.error,
                model_tag=model_tag,
                messages=test_result.result.final_messages
                if test_result.result is not None
                else None,
                row_index=row_index,
            )

        # Apply eval functions
        messages = test_result.result.final_messages
        ground_truth = row["ground_truth"]
        metadata = dict(row)

        scores: dict[str, float] = {}
        for eval_fn in self.eval_fns:
            try:
                score = await eval_fn(messages, ground_truth, metadata)
                scores[eval_fn.name] = score
            except Exception as e:
                logger.warning(
                    "Eval function '%s' failed on row %d run %d: %s",
                    eval_fn.name,
                    row_index,
                    run_index,
                    e,
                )
                scores[eval_fn.name] = 0.0

        return EvalRunResult(
            run_index=run_index,
            success=True,
            scores=scores,
            duration_ms=test_result.duration_ms,
            tokens=test_result.token_usage.get("total_tokens", 0),
            model_tag=model_tag,
            messages=test_result.result.final_messages,
            row_index=row_index,
        )

    async def run_batch(
        self,
        work_items: list[tuple[DatasetRow, int, int, str | None]],
        max_turns: int = 10,
        completion_params: dict[str, Any] | None = None,
    ) -> tuple[list[EvalRunResult | None], str | None]:
        """Run a batch of work items concurrently and return ordered results.

        Args:
            work_items: List of (row, row_index, run_index, model_tag) tuples.
            max_turns: Max agent turns per run.
            completion_params: LLM sampling parameters.

        Returns:
            A tuple of (batch_results, systemic_error) where batch_results is
            a list matching the order of work_items (None for items not
            attempted), and systemic_error is the error string if a
            SystemicProviderError occurred.
        """
        if not work_items:
            return [], None

        pool_size = len(work_items)

        # Build runner pools
        has_baseline_items = any(tag == "baseline" for _, _, _, tag in work_items)
        if has_baseline_items:
            primary_pool_size = max(1, (pool_size + 1) // 2)
            baseline_pool_size = max(1, pool_size - primary_pool_size)
        else:
            primary_pool_size = pool_size
            baseline_pool_size = 0

        primary_runners: list[LocalRolloutRunner] = []
        baseline_runners: list[LocalRolloutRunner] = []
        try:
            for _ in range(primary_pool_size):
                primary_runners.append(self._create_rollout_runner())
            for _ in range(baseline_pool_size):
                baseline_runners.append(self._create_baseline_rollout_runner())
        except Exception:
            # Clean up already-created runners
            for runner in primary_runners + baseline_runners:
                close = getattr(runner.llm_client, "close", None)
                if callable(close):
                    try:
                        maybe_awaitable = close()
                        if inspect.isawaitable(maybe_awaitable):
                            await maybe_awaitable
                    except Exception:
                        pass
            raise

        primary_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        for runner in primary_runners:
            primary_pool.put_nowait(runner)

        baseline_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        for runner in baseline_runners:
            baseline_pool.put_nowait(runner)

        all_runners = primary_runners + baseline_runners

        batch_results: list[EvalRunResult | None] = [None] * len(work_items)
        systemic_error: str | None = None

        async def _close_runner_pool() -> None:
            """Best-effort cleanup for per-runner LLM clients."""
            for runner in all_runners:
                close = getattr(runner.llm_client, "close", None)
                if not callable(close):
                    continue
                try:
                    maybe_awaitable = close()
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                except Exception:
                    logger.debug(
                        "Failed to close concurrent runner client", exc_info=True
                    )

        async def _run_one(
            idx: int,
            row: DatasetRow,
            row_index: int,
            run_index: int,
            model_tag: str | None,
        ) -> None:
            nonlocal systemic_error
            runner: LocalRolloutRunner | None = None
            target_pool = baseline_pool if model_tag == "baseline" else primary_pool
            run_start = time.monotonic()
            try:
                runner = await target_pool.get()
                run_start = time.monotonic()
                result = await self.run_single(
                    row=row,
                    row_index=row_index,
                    run_index=run_index,
                    max_turns=max_turns,
                    completion_params=completion_params,
                    runner=runner,
                    model_tag=model_tag,
                )
                batch_results[idx] = result
            except SystemicProviderError as e:
                duration_ms, tokens = _extract_systemic_error_metrics(
                    e,
                    fallback_started_at=run_start,
                )
                batch_results[idx] = EvalRunResult(
                    run_index=run_index,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    tokens=tokens,
                    model_tag=model_tag,
                    messages=None,
                    row_index=row_index,
                )
                if systemic_error is None:
                    systemic_error = str(e)
            finally:
                if runner is not None:
                    target_pool.put_nowait(runner)

        try:
            tasks: list[asyncio.Task[None]] = [
                asyncio.create_task(_run_one(idx, row, row_index, run_index, model_tag))
                for idx, (row, row_index, run_index, model_tag) in enumerate(work_items)
            ]

            try:
                # Wait for all tasks — let them all complete even if
                # a systemic error occurs, since they're already in-flight.
                for task in asyncio.as_completed(tasks):
                    await task
            except Exception:
                # Unexpected error: cancel remaining in-flight tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            await _close_runner_pool()

        return batch_results, systemic_error

    async def run_eval(
        self,
        rows: list[DatasetRow],
        n_runs: int = 1,
        max_turns: int = 10,
        completion_params: dict[str, Any] | None = None,
        pass_threshold: float = 1.0,
        on_progress: Callable[[int, int, EvalRunResult], None] | None = None,
        start_index: int = 0,
        batch_size: int = 1,
    ) -> EvalResult:
        """Run the full evaluation.

        Args:
            rows: Dataset rows.
            n_runs: Number of runs per row (for pass@n).
            max_turns: Max agent turns per run.
            completion_params: LLM sampling parameters.
            pass_threshold: Score >= this counts as pass for pass@k.
            on_progress: Callback after each run: (current, total, result).
            start_index: Starting row index for numbering.
            batch_size: Number of concurrent runs. Default 1 (sequential).
                For ``batch_size > 1``, runs execute in waves of at most
                ``batch_size``. If any run in a wave fails, remaining waves
                are skipped.

        Returns:
            EvalResult with all results and statistics.
        """
        if batch_size > 1:
            return await self._run_eval_concurrent(
                rows=rows,
                n_runs=n_runs,
                max_turns=max_turns,
                completion_params=completion_params,
                pass_threshold=pass_threshold,
                on_progress=on_progress,
                start_index=start_index,
                batch_size=batch_size,
            )

        total_start = time.monotonic()
        row_results: list[EvalRowResult] = []
        model_tags: list[str | None] = (
            ["primary", "baseline"] if self.has_baseline else [None]
        )
        total = len(rows) * n_runs * len(model_tags)
        current = 0
        stopped_early = False
        stop_reason: str | None = None

        for i, row in enumerate(rows):
            row_index = start_index + i
            row_result = EvalRowResult(row_index=row_index)

            for run_idx in range(n_runs):
                for tag in model_tags:
                    run_start = time.monotonic()
                    try:
                        result = await self.run_single(
                            row=row,
                            row_index=row_index,
                            run_index=run_idx,
                            max_turns=max_turns,
                            completion_params=completion_params,
                            model_tag=tag,
                        )
                    except SystemicProviderError as e:
                        duration_ms, tokens = _extract_systemic_error_metrics(
                            e,
                            fallback_started_at=run_start,
                        )
                        result = EvalRunResult(
                            run_index=run_idx,
                            success=False,
                            error=str(e),
                            duration_ms=duration_ms,
                            tokens=tokens,
                            model_tag=tag,
                            messages=None,
                            row_index=row_index,
                        )
                        row_result.runs.append(result)
                        current += 1
                        if on_progress:
                            on_progress(current, total, result)
                        if current < total:
                            stopped_early = True
                            stop_reason = str(e)
                        break

                    row_result.runs.append(result)
                    current += 1

                    if on_progress:
                        on_progress(current, total, result)

                if stopped_early:
                    break

            row_results.append(row_result)
            if stopped_early:
                break

        total_duration_ms = (time.monotonic() - total_start) * 1000
        total_tokens = sum(run.tokens for row in row_results for run in row.runs)

        # Compute per-model summaries if baseline is configured
        model_summaries: list[EvalModelSummary] | None = None
        if self.has_baseline:
            model_summaries = self._compute_model_summaries(
                row_results, n_runs, pass_threshold
            )

        # Top-level eval summaries: primary model only when baseline is present,
        # so that the "official" summary is never polluted by baseline scores.
        if self.has_baseline:
            primary_rows = self._filter_runs_by_tag(row_results, "primary")
            eval_summaries = self._compute_summaries(
                primary_rows, n_runs, pass_threshold
            )
        else:
            eval_summaries = self._compute_summaries(
                row_results, n_runs, pass_threshold
            )

        return EvalResult(
            rows=row_results,
            eval_summaries=eval_summaries,
            total_rows=len(row_results),
            total_runs=current,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            n_runs=n_runs,
            pass_threshold=pass_threshold,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            model_summaries=model_summaries,
        )

    async def _run_eval_concurrent(
        self,
        rows: list[DatasetRow],
        n_runs: int,
        max_turns: int,
        completion_params: dict[str, Any] | None,
        pass_threshold: float,
        on_progress: Callable[[int, int, EvalRunResult], None] | None,
        start_index: int,
        batch_size: int,
    ) -> EvalResult:
        """Run evaluation with concurrent execution.

        Creates a pool of LocalRolloutRunner instances (each with its own
        ExternalLLMClient) and dispatches runs concurrently, limited by
        batch_size.  When a baseline model is configured, two separate pools
        are created and the batch_size is split between them.
        """
        total_start = time.monotonic()
        model_tags: list[str | None] = (
            ["primary", "baseline"] if self.has_baseline else [None]
        )
        total = len(rows) * n_runs * len(model_tags)
        pool_size = min(batch_size, total)

        # Build runner pools.  When baseline is configured, size pools to match
        # the expected task distribution: work items alternate [primary, baseline,
        # primary, ...], so each batch of N items contains ceil(N/2) primary and
        # floor(N/2) baseline tasks.
        if self.has_baseline:
            primary_pool_size = max(1, (pool_size + 1) // 2)
            baseline_pool_size = max(1, pool_size - primary_pool_size)
        else:
            primary_pool_size = pool_size
            baseline_pool_size = 0

        primary_runners = [
            self._create_rollout_runner() for _ in range(primary_pool_size)
        ]
        primary_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        for runner in primary_runners:
            primary_pool.put_nowait(runner)

        baseline_runners: list[LocalRolloutRunner] = []
        baseline_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        if baseline_pool_size > 0:
            baseline_runners = [
                self._create_baseline_rollout_runner()
                for _ in range(baseline_pool_size)
            ]
            for runner in baseline_runners:
                baseline_pool.put_nowait(runner)

        all_runners = primary_runners + baseline_runners

        completed = 0
        stopped_early = False
        stop_reason: str | None = None
        completed_results: list[tuple[int, EvalRunResult]] = []

        async def _close_runner_pool() -> None:
            """Best-effort cleanup for per-runner LLM clients."""
            for runner in all_runners:
                close = getattr(runner.llm_client, "close", None)
                if not callable(close):
                    continue
                try:
                    maybe_awaitable = close()
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                except Exception:
                    logger.debug(
                        "Failed to close concurrent runner client", exc_info=True
                    )

        async def _run_one(
            row: DatasetRow,
            row_index: int,
            run_index: int,
            model_tag: str | None,
        ) -> tuple[int, EvalRunResult]:
            nonlocal completed
            runner: LocalRolloutRunner | None = None
            target_pool = baseline_pool if model_tag == "baseline" else primary_pool
            run_start = time.monotonic()
            try:
                runner = await target_pool.get()
                run_start = time.monotonic()
                result = await self.run_single(
                    row=row,
                    row_index=row_index,
                    run_index=run_index,
                    max_turns=max_turns,
                    completion_params=completion_params,
                    runner=runner,
                    model_tag=model_tag,
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total, result)
                completed_results.append((row_index, result))
                return row_index, result
            except SystemicProviderError as e:
                # Record the failed result, then re-raise so the batch
                # loop can detect the systemic failure and stop early.
                duration_ms, tokens = _extract_systemic_error_metrics(
                    e,
                    fallback_started_at=run_start,
                )
                result = EvalRunResult(
                    run_index=run_index,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    tokens=tokens,
                    model_tag=model_tag,
                    messages=None,
                    row_index=row_index,
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total, result)
                completed_results.append((row_index, result))
                raise
            finally:
                if runner is not None:
                    target_pool.put_nowait(runner)

        # Build work items: interleave primary/baseline per (row, run_idx)
        work_items: list[tuple[DatasetRow, int, int, str | None]] = []
        for i, row in enumerate(rows):
            row_index = start_index + i
            for run_idx in range(n_runs):
                for tag in model_tags:
                    work_items.append((row, row_index, run_idx, tag))

        try:
            cursor = 0
            while cursor < len(work_items):
                batch = work_items[cursor : cursor + batch_size]
                tasks: list[asyncio.Task[tuple[int, EvalRunResult]]] = [
                    asyncio.create_task(_run_one(row, row_index, run_idx, tag))
                    for row, row_index, run_idx, tag in batch
                ]

                systemic_error: str | None = None

                try:
                    for task in asyncio.as_completed(tasks):
                        try:
                            await task
                        except SystemicProviderError as e:
                            # Record error but keep awaiting remaining
                            # tasks in this batch so their results are collected.
                            if systemic_error is None:
                                systemic_error = str(e)
                except Exception:
                    # Unexpected error: cancel remaining in-flight tasks
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

                cursor += len(batch)

                if systemic_error is not None:
                    if cursor < len(work_items):
                        stopped_early = True
                        stop_reason = systemic_error
                    break
        finally:
            await _close_runner_pool()

        # Organise flat results back into per-row structure.
        row_results_map: dict[int, EvalRowResult] = {}
        for row_index, run_result in completed_results:
            if row_index not in row_results_map:
                row_results_map[row_index] = EvalRowResult(row_index=row_index)
            row_results_map[row_index].runs.append(run_result)

        # Ensure deterministic ordering within each row:
        # sort by (model_tag, run_index) so primary runs come first.
        tag_order = {"primary": 0, "baseline": 1}
        for row_result in row_results_map.values():
            row_result.runs.sort(
                key=lambda r: (tag_order.get(r.model_tag or "", -1), r.run_index)
            )

        row_results = [row_results_map[i] for i in sorted(row_results_map.keys())]

        total_duration_ms = (time.monotonic() - total_start) * 1000
        total_tokens = sum(run.tokens for row in row_results for run in row.runs)

        # Compute per-model summaries if baseline is configured
        model_summaries: list[EvalModelSummary] | None = None
        if self.has_baseline:
            model_summaries = self._compute_model_summaries(
                row_results, n_runs, pass_threshold
            )

        # Top-level eval summaries: primary model only when baseline is present,
        # so that the "official" summary is never polluted by baseline scores.
        if self.has_baseline:
            primary_rows = self._filter_runs_by_tag(row_results, "primary")
            eval_summaries = self._compute_summaries(
                primary_rows, n_runs, pass_threshold
            )
        else:
            eval_summaries = self._compute_summaries(
                row_results, n_runs, pass_threshold
            )

        return EvalResult(
            rows=row_results,
            eval_summaries=eval_summaries,
            total_rows=len(row_results),
            total_runs=len(completed_results),
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            n_runs=n_runs,
            pass_threshold=pass_threshold,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            model_summaries=model_summaries,
        )

    def _compute_summaries(
        self,
        row_results: list[EvalRowResult],
        n_runs: int,
        pass_threshold: float,
    ) -> dict[str, EvalEvalSummary]:
        """Compute per-eval-function summary statistics.

        Failed runs and missing eval scores are treated as 0.0 so reliability
        issues are reflected in evaluation quality metrics.
        """
        import math

        summaries: dict[str, EvalEvalSummary] = {}

        for eval_fn in self.eval_fns:
            name = eval_fn.name

            # Collect all scores for this eval fn. Missing scores count as 0.
            all_scores: list[float] = []
            for row in row_results:
                for run in row.runs:
                    all_scores.append(run.scores.get(name, 0.0))

            if not all_scores:
                summaries[name] = EvalEvalSummary()
                continue

            mean = sum(all_scores) / len(all_scores)
            variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            std = math.sqrt(variance)

            summary = EvalEvalSummary(
                mean=mean,
                std=std,
                min=min(all_scores),
                max=max(all_scores),
            )

            # Compute pass@k if n > 1
            if n_runs > 1:
                from osmosis_ai.rollout.eval.evaluation.report import pass_at_k

                for k in [1, 3, 5, 10]:
                    if k > n_runs:
                        break
                    # Compute pass@k per row, then average.
                    # Use n_runs (not len(row.runs)) so that missing runs
                    # (e.g. from early stopping) are treated as failures.
                    row_pass_at_k: list[float] = []
                    for row in row_results:
                        c = sum(
                            1
                            for run in row.runs
                            if run.scores.get(name, 0.0) >= pass_threshold
                        )
                        n = max(len(row.runs), n_runs)
                        if n > 0:
                            row_pass_at_k.append(pass_at_k(n, c, k))
                    if row_pass_at_k:
                        summary.pass_at_k[k] = sum(row_pass_at_k) / len(row_pass_at_k)

            summaries[name] = summary

        return summaries

    def _filter_runs_by_tag(
        self,
        row_results: list[EvalRowResult],
        model_tag: str,
    ) -> list[EvalRowResult]:
        """Build row results containing only runs matching the given model_tag."""
        filtered: list[EvalRowResult] = []
        for row in row_results:
            tagged_runs = [r for r in row.runs if r.model_tag == model_tag]
            if tagged_runs:
                filtered.append(
                    EvalRowResult(
                        row_index=row.row_index,
                        runs=tagged_runs,
                    )
                )
        return filtered

    def _compute_model_summaries(
        self,
        row_results: list[EvalRowResult],
        n_runs: int,
        pass_threshold: float,
    ) -> list[EvalModelSummary]:
        """Compute per-model summary statistics for comparison mode."""
        summaries: list[EvalModelSummary] = []
        for tag in ("primary", "baseline"):
            filtered = self._filter_runs_by_tag(row_results, tag)
            eval_summaries = self._compute_summaries(filtered, n_runs, pass_threshold)
            all_runs = [run for row in filtered for run in row.runs]
            model_name = ""
            if tag == "primary":
                model_name = self.llm_client.display_name
            elif self._baseline_llm_client is not None:
                model_name = self._baseline_llm_client.display_name
            summaries.append(
                EvalModelSummary(
                    model=model_name,
                    model_tag=tag,
                    eval_summaries=eval_summaries,
                    total_runs=len(all_runs),
                    total_tokens=sum(r.tokens for r in all_runs),
                    total_duration_ms=sum(r.duration_ms for r in all_runs),
                )
            )
        return summaries


__all__ = [
    "EvalEvalSummary",
    "EvalModelSummary",
    "EvalResult",
    "EvalRowResult",
    "EvalRunResult",
    "EvalRunner",
]
