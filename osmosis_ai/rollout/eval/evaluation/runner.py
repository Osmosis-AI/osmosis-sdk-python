"""Evaluation runner for executing agent loops with eval functions.

Runs the agent against each dataset row (optionally multiple times for pass@n),
applies eval functions to each result, and aggregates statistics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.eval.common.dataset import DatasetRow
from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient
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
    """

    rows: List[EvalRowResult]
    eval_summaries: Dict[str, EvalEvalSummary]
    total_rows: int
    total_runs: int
    total_tokens: int
    total_duration_ms: float
    n_runs: int
    pass_threshold: float


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

    def _default_llm_client_factory(self) -> ExternalLLMClient:
        """Create a new ExternalLLMClient from the original client's config."""
        return ExternalLLMClient(
            model=self.llm_client.model,
            api_key=self.llm_client._api_key,
            api_base=self.llm_client._api_base,
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

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        run_index: int,
        max_turns: int = 10,
        completion_params: Optional[Dict[str, Any]] = None,
        runner: Optional[LocalRolloutRunner] = None,
    ) -> EvalRunResult:
        """Run agent once on a row and apply all eval functions.

        Args:
            row: Dataset row.
            row_index: Index of the row in the dataset.
            run_index: Which run this is (for pass@n).
            max_turns: Max agent turns.
            completion_params: LLM sampling parameters.
            runner: Optional runner instance for concurrent execution.

        Returns:
            EvalRunResult with scores from all eval functions.
        """
        # Run the agent
        rollout_runner = runner or self._rollout_runner
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
        total = len(rows) * n_runs
        current = 0

        for i, row in enumerate(rows):
            row_index = start_index + i
            row_result = EvalRowResult(row_index=row_index)

            for run_idx in range(n_runs):
                result = await self.run_single(
                    row=row,
                    row_index=row_index,
                    run_index=run_idx,
                    max_turns=max_turns,
                    completion_params=completion_params,
                )
                row_result.runs.append(result)
                current += 1

                if on_progress:
                    on_progress(current, total, result)

            row_results.append(row_result)

        total_duration_ms = (time.monotonic() - total_start) * 1000
        total_tokens = sum(
            run.tokens for row in row_results for run in row.runs
        )

        # Compute eval summaries
        eval_summaries = self._compute_summaries(
            row_results, n_runs, pass_threshold
        )

        return EvalResult(
            rows=row_results,
            eval_summaries=eval_summaries,
            total_rows=len(rows),
            total_runs=current,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            n_runs=n_runs,
            pass_threshold=pass_threshold,
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
        batch_size.
        """
        total_start = time.monotonic()
        total = len(rows) * n_runs
        pool_size = min(batch_size, total)

        # Build a runner pool — each runner has an independent LLM client
        # so metrics and tool state don't collide across concurrent runs.
        pool: asyncio.Queue[LocalRolloutRunner] = asyncio.Queue()
        for _ in range(pool_size):
            pool.put_nowait(self._create_rollout_runner())

        completed = 0

        async def _run_one(
            row: DatasetRow, row_index: int, run_index: int,
        ) -> EvalRunResult:
            nonlocal completed
            runner = await pool.get()
            try:
                result = await self.run_single(
                    row=row,
                    row_index=row_index,
                    run_index=run_index,
                    max_turns=max_turns,
                    completion_params=completion_params,
                    runner=runner,
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total, result)
                return result
            finally:
                pool.put_nowait(runner)

        # Launch all runs; the pool size naturally limits concurrency.
        coros = []
        for i, row in enumerate(rows):
            row_index = start_index + i
            for run_idx in range(n_runs):
                coros.append(_run_one(row, row_index, run_idx))

        results = await asyncio.gather(*coros)

        # Organise flat results back into per-row structure.
        row_results_map: Dict[int, EvalRowResult] = {}
        for idx, (i, row) in enumerate(
            (i, row)
            for i, row in enumerate(rows)
            for _ in range(n_runs)
        ):
            row_index = start_index + i
            if row_index not in row_results_map:
                row_results_map[row_index] = EvalRowResult(row_index=row_index)
            row_results_map[row_index].runs.append(results[idx])

        # Ensure deterministic ordering within each row.
        for row_result in row_results_map.values():
            row_result.runs.sort(key=lambda r: r.run_index)

        row_results = [row_results_map[start_index + i] for i in range(len(rows))]

        total_duration_ms = (time.monotonic() - total_start) * 1000
        total_tokens = sum(
            run.tokens for row in row_results for run in row.runs
        )

        eval_summaries = self._compute_summaries(
            row_results, n_runs, pass_threshold
        )

        return EvalResult(
            rows=row_results,
            eval_summaries=eval_summaries,
            total_rows=len(rows),
            total_runs=total,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            n_runs=n_runs,
            pass_threshold=pass_threshold,
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

__all__ = [
    "EvalEvalSummary",
    "EvalResult",
    "EvalRowResult",
    "EvalRunResult",
    "EvalRunner",
]
