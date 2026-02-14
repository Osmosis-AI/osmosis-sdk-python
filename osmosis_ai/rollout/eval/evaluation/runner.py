"""Evaluation runner for executing agent loops with eval functions.

Runs the agent against each dataset row (optionally multiple times for pass@n),
applies eval functions to each result, and aggregates statistics.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.eval.common.dataset import DatasetRow
from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
from osmosis_ai.rollout.eval.common.runner import LocalRolloutRunner

logger = logging.getLogger(__name__)


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
    """

    run_index: int
    success: bool
    scores: Dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    tokens: int = 0
    error: Optional[str] = None
    model_tag: Optional[str] = None


@dataclass
class EvalRowResult:
    """Results from all runs of a single dataset row.

    Attributes:
        row_index: Index of the row in the dataset.
        runs: List of run results.
    """

    row_index: int
    runs: List[EvalRunResult] = field(default_factory=list)


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
    pass_at_k: Dict[int, float] = field(default_factory=dict)


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
    eval_summaries: Dict[str, EvalEvalSummary]
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

    rows: List[EvalRowResult]
    eval_summaries: Dict[str, EvalEvalSummary]
    total_rows: int
    total_runs: int
    total_tokens: int
    total_duration_ms: float
    n_runs: int
    pass_threshold: float
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    model_summaries: Optional[List[EvalModelSummary]] = None


class EvalRunner:
    """Orchestrates evaluation execution: agent runs + eval function scoring."""

    def __init__(
        self,
        agent_loop: RolloutAgentLoop,
        llm_client: ExternalLLMClient,
        eval_fns: List[EvalFnWrapper],
        debug: bool = False,
        debug_dir: Optional[str] = None,
        llm_client_factory: Optional[Callable[[], Any]] = None,
        baseline_llm_client: Optional[ExternalLLMClient] = None,
        baseline_llm_client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.llm_client = llm_client
        self.eval_fns = eval_fns
        self.debug = debug
        self.debug_dir = debug_dir
        self._llm_client_factory = llm_client_factory or self._default_llm_client_factory
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
        self._baseline_rollout_runner: Optional[LocalRolloutRunner] = None
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
            api_key=self.llm_client._api_key,
            api_base=self.llm_client._api_base,
        )

    def _default_baseline_llm_client_factory(self) -> ExternalLLMClient:
        """Create a new ExternalLLMClient from the baseline client's config."""
        if self._baseline_llm_client is None:
            raise RuntimeError("No baseline LLM client configured")
        return ExternalLLMClient(
            model=self._baseline_llm_client.display_name,
            api_key=self._baseline_llm_client._api_key,
            api_base=self._baseline_llm_client._api_base,
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
        completion_params: Optional[Dict[str, Any]] = None,
        runner: Optional[LocalRolloutRunner] = None,
        model_tag: Optional[str] = None,
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
        test_result = await rollout_runner.run_single(
            row=row,
            row_index=row_index,
            max_turns=max_turns,
            completion_params=completion_params,
            rollout_id=f"eval-{row_index}-run-{run_index}",
            request_metadata={"run_index": run_index},
        )

        if not test_result.success or test_result.result is None:
            return EvalRunResult(
                run_index=run_index,
                success=False,
                duration_ms=test_result.duration_ms,
                tokens=test_result.token_usage.get("total_tokens", 0),
                error=test_result.error,
                model_tag=model_tag,
            )

        # Apply eval functions
        messages = test_result.result.final_messages
        ground_truth = row["ground_truth"]
        metadata = dict(row)

        scores: Dict[str, float] = {}
        for eval_fn in self.eval_fns:
            try:
                score = await eval_fn(messages, ground_truth, metadata)
                scores[eval_fn.name] = score
            except Exception as e:
                logger.warning(
                    "Eval function '%s' failed on row %d run %d: %s",
                    eval_fn.name, row_index, run_index, e,
                )
                scores[eval_fn.name] = 0.0

        return EvalRunResult(
            run_index=run_index,
            success=True,
            scores=scores,
            duration_ms=test_result.duration_ms,
            tokens=test_result.token_usage.get("total_tokens", 0),
            model_tag=model_tag,
        )

    async def run_eval(
        self,
        rows: List[DatasetRow],
        n_runs: int = 1,
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        pass_threshold: float = 1.0,
        on_progress: Optional[Callable[[int, int, EvalRunResult], None]] = None,
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
        row_results: List[EvalRowResult] = []
        model_tags = (
            ["primary", "baseline"] if self.has_baseline else [None]
        )
        total = len(rows) * n_runs * len(model_tags)
        current = 0
        stopped_early = False
        stop_reason: Optional[str] = None

        for i, row in enumerate(rows):
            row_index = start_index + i
            row_result = EvalRowResult(row_index=row_index)

            for run_idx in range(n_runs):
                for tag in model_tags:
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
                        result = EvalRunResult(
                            run_index=run_idx,
                            success=False,
                            error=str(e),
                            model_tag=tag,
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

                    if not result.success:
                        if current < total:
                            stopped_early = True
                            stop_reason = result.error
                        break

                if stopped_early:
                    break

            row_results.append(row_result)
            if stopped_early:
                break

        total_duration_ms = (time.monotonic() - total_start) * 1000
        total_tokens = sum(
            run.tokens for row in row_results for run in row.runs
        )

        # Compute per-model summaries if baseline is configured
        model_summaries: Optional[List[EvalModelSummary]] = None
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
        rows: List[DatasetRow],
        n_runs: int,
        max_turns: int,
        completion_params: Optional[Dict[str, Any]],
        pass_threshold: float,
        on_progress: Optional[Callable[[int, int, EvalRunResult], None]],
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
        model_tags: List[Optional[str]] = (
            ["primary", "baseline"] if self.has_baseline else [None]
        )
        total = len(rows) * n_runs * len(model_tags)
        pool_size = min(batch_size, total)

        # Build runner pools.  When baseline is configured, split slots
        # between primary and baseline (each gets at least 1).
        if self.has_baseline:
            primary_pool_size = max(1, pool_size // 2)
            baseline_pool_size = max(1, pool_size - primary_pool_size)
        else:
            primary_pool_size = pool_size
            baseline_pool_size = 0

        primary_runners = [self._create_rollout_runner() for _ in range(primary_pool_size)]
        primary_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        for runner in primary_runners:
            primary_pool.put_nowait(runner)

        baseline_runners: List[LocalRolloutRunner] = []
        baseline_pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        if baseline_pool_size > 0:
            baseline_runners = [self._create_baseline_rollout_runner() for _ in range(baseline_pool_size)]
            for runner in baseline_runners:
                baseline_pool.put_nowait(runner)

        all_runners = primary_runners + baseline_runners

        completed = 0
        stopped_early = False
        stop_reason: Optional[str] = None
        completed_results: List[Tuple[int, EvalRunResult]] = []

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
                    logger.debug("Failed to close concurrent runner client", exc_info=True)

        async def _run_one(
            row: DatasetRow, row_index: int, run_index: int,
            model_tag: Optional[str],
        ) -> Tuple[int, EvalRunResult]:
            nonlocal completed
            runner: Optional[LocalRolloutRunner] = None
            target_pool = baseline_pool if model_tag == "baseline" else primary_pool
            try:
                runner = await target_pool.get()
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
                result = EvalRunResult(
                    run_index=run_index,
                    success=False,
                    error=str(e),
                    model_tag=model_tag,
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total, result)
                completed_results.append((row_index, result))
                return row_index, result
            finally:
                if runner is not None:
                    target_pool.put_nowait(runner)

        # Build work items: interleave primary/baseline per (row, run_idx)
        work_items: List[Tuple[DatasetRow, int, int, Optional[str]]] = []
        for i, row in enumerate(rows):
            row_index = start_index + i
            for run_idx in range(n_runs):
                for tag in model_tags:
                    work_items.append((row, row_index, run_idx, tag))

        try:
            cursor = 0
            while cursor < len(work_items):
                batch = work_items[cursor:cursor + batch_size]
                tasks: List[asyncio.Task[Tuple[int, EvalRunResult]]] = [
                    asyncio.create_task(_run_one(row, row_index, run_idx, tag))
                    for row, row_index, run_idx, tag in batch
                ]

                batch_failed = False
                batch_error: Optional[str] = None

                for task in asyncio.as_completed(tasks):
                    _row_index, result = await task
                    if not result.success and not batch_failed:
                        batch_failed = True
                        batch_error = result.error

                await asyncio.gather(*tasks, return_exceptions=True)

                cursor += len(batch)

                if batch_failed:
                    if cursor < len(work_items):
                        stopped_early = True
                        stop_reason = batch_error
                    break
        except Exception:
            raise
        finally:
            await _close_runner_pool()

        # Organise flat results back into per-row structure.
        row_results_map: Dict[int, EvalRowResult] = {}
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
        total_tokens = sum(
            run.tokens for row in row_results for run in row.runs
        )

        # Compute per-model summaries if baseline is configured
        model_summaries: Optional[List[EvalModelSummary]] = None
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
        row_results: List[EvalRowResult],
        n_runs: int,
        pass_threshold: float,
    ) -> Dict[str, EvalEvalSummary]:
        """Compute per-eval-function summary statistics.

        Failed runs and missing eval scores are treated as 0.0 so reliability
        issues are reflected in evaluation quality metrics.
        """
        import math

        summaries: Dict[str, EvalEvalSummary] = {}

        for eval_fn in self.eval_fns:
            name = eval_fn.name

            # Collect all scores for this eval fn. Missing scores count as 0.
            all_scores: List[float] = []
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
                    # Compute pass@k per row, then average
                    row_pass_at_k: List[float] = []
                    for row in row_results:
                        c = sum(
                            1 for run in row.runs
                            if run.scores.get(name, 0.0) >= pass_threshold
                        )
                        n = len(row.runs)
                        if n > 0:
                            row_pass_at_k.append(pass_at_k(n, c, k))
                    if row_pass_at_k:
                        summary.pass_at_k[k] = sum(row_pass_at_k) / len(row_pass_at_k)

            summaries[name] = summary

        return summaries

    def _filter_runs_by_tag(
        self,
        row_results: List[EvalRowResult],
        model_tag: str,
    ) -> List[EvalRowResult]:
        """Build row results containing only runs matching the given model_tag."""
        filtered: List[EvalRowResult] = []
        for row in row_results:
            tagged_runs = [r for r in row.runs if r.model_tag == model_tag]
            if tagged_runs:
                filtered.append(EvalRowResult(
                    row_index=row.row_index,
                    runs=tagged_runs,
                ))
        return filtered

    def _compute_model_summaries(
        self,
        row_results: List[EvalRowResult],
        n_runs: int,
        pass_threshold: float,
    ) -> List[EvalModelSummary]:
        """Compute per-model summary statistics for comparison mode."""
        summaries: List[EvalModelSummary] = []
        for tag in ("primary", "baseline"):
            filtered = self._filter_runs_by_tag(row_results, tag)
            eval_summaries = self._compute_summaries(filtered, n_runs, pass_threshold)
            all_runs = [run for row in filtered for run in row.runs]
            model_name = ""
            if tag == "primary":
                model_name = self.llm_client.display_name
            elif self._baseline_llm_client is not None:
                model_name = self._baseline_llm_client.display_name
            summaries.append(EvalModelSummary(
                model=model_name,
                model_tag=tag,
                eval_summaries=eval_summaries,
                total_runs=len(all_runs),
                total_tokens=sum(r.tokens for r in all_runs),
                total_duration_ms=sum(r.duration_ms for r in all_runs),
            ))
        return summaries

__all__ = [
    "EvalEvalSummary",
    "EvalModelSummary",
    "EvalResult",
    "EvalRowResult",
    "EvalRunResult",
    "EvalRunner",
]
