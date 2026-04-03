"""Evaluation runner for executing workflows with eval functions.

Runs the workflow against each dataset row (optionally multiple times for pass@n),
applies eval functions to each result, and aggregates statistics.

Uses semaphore-based concurrency via WorkflowExecutor (v2).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.eval.common.dataset import DatasetRow, dataset_row_to_prompt
from osmosis_ai.eval.common.errors import SystemicProviderError
from osmosis_ai.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.eval.executor import WorkflowExecutor

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
        median: Median score.
        std: Standard deviation of scores.
        min: Minimum score.
        max: Maximum score.
        p25: 25th percentile score.
        p75: 75th percentile score.
        pass_at_k: Dict of k -> pass@k value. Only populated when n > 1.
    """

    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
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
    """Orchestrates evaluation execution: workflow runs + eval function scoring.

    Uses semaphore-based concurrency with WorkflowExecutor instead of runner pools.
    """

    def __init__(
        self,
        executor: WorkflowExecutor,
        eval_fns: list[EvalFnWrapper],
        baseline_executor: WorkflowExecutor | None = None,
        primary_model_name: str = "",
        baseline_model_name: str = "",
    ) -> None:
        self.executor = executor
        self.eval_fns = eval_fns
        self.baseline_executor = baseline_executor
        self.primary_model_name = primary_model_name
        self.baseline_model_name = baseline_model_name

    @property
    def has_baseline(self) -> bool:
        """Whether a baseline executor is configured for comparison."""
        return self.baseline_executor is not None

    @staticmethod
    def _extract_messages(samples: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract messages from the first sample."""
        for sample in samples.values():
            msgs = getattr(sample, "messages", None)
            if msgs is None and isinstance(sample, dict):
                msgs = sample.get("messages", [])
            if msgs:
                return list(msgs)
        return []

    async def run_single(
        self,
        row: DatasetRow,
        row_index: int,
        run_index: int,
        model_tag: str | None = None,
    ) -> EvalRunResult:
        """Run workflow once on a row and apply all eval functions."""
        prompt = dataset_row_to_prompt(row)
        rollout_id = f"eval-{row_index}-run-{run_index}"
        if model_tag:
            rollout_id += f"-{model_tag}"

        executor = (
            self.baseline_executor
            if model_tag == "baseline" and self.baseline_executor
            else self.executor
        )

        run_start = time.monotonic()
        try:
            result = await executor.run_single(prompt, rollout_id)
        except SystemicProviderError as e:
            duration_ms, tokens = _extract_systemic_error_metrics(
                e, fallback_started_at=run_start
            )
            e.duration_ms = duration_ms
            e.tokens = tokens
            raise

        if not result.success:
            return EvalRunResult(
                run_index=run_index,
                success=False,
                duration_ms=result.duration_ms,
                tokens=result.metrics.total_tokens,
                error=result.error,
                model_tag=model_tag,
                messages=None,
                row_index=row_index,
            )

        messages = self._extract_messages(result.samples)

        ground_truth = row["ground_truth"]
        metadata = dict(row)

        async def _run_eval_fn(fn: EvalFnWrapper) -> tuple[str, float]:
            try:
                return fn.name, await fn(messages, ground_truth, metadata)
            except Exception as e:
                logger.warning(
                    "Eval function '%s' failed on row %d run %d: %s",
                    fn.name,
                    row_index,
                    run_index,
                    e,
                )
                return fn.name, 0.0

        score_pairs = await asyncio.gather(*[_run_eval_fn(fn) for fn in self.eval_fns])
        scores: dict[str, float] = dict(score_pairs)

        return EvalRunResult(
            run_index=run_index,
            success=True,
            scores=scores,
            duration_ms=result.duration_ms,
            tokens=result.metrics.total_tokens,
            model_tag=model_tag,
            messages=messages,
            row_index=row_index,
        )

    async def run_batch(
        self,
        work_items: list[tuple[DatasetRow, int, int, str | None]],
        batch_size: int = 4,
    ) -> tuple[list[EvalRunResult | None], str | None]:
        """Run work items concurrently with a semaphore.

        Returns:
            A tuple of (batch_results, systemic_error) where batch_results is
            a list matching the order of work_items (None for items not
            attempted), and systemic_error is the error string if a
            SystemicProviderError occurred.
        """
        if not work_items:
            return [], None

        semaphore = asyncio.Semaphore(batch_size)
        batch_results: list[EvalRunResult | None] = [None] * len(work_items)
        systemic_error: str | None = None

        async def _run_one(
            idx: int,
            row: DatasetRow,
            row_index: int,
            run_index: int,
            model_tag: str | None,
        ) -> None:
            nonlocal systemic_error
            async with semaphore:
                try:
                    result = await self.run_single(
                        row=row,
                        row_index=row_index,
                        run_index=run_index,
                        model_tag=model_tag,
                    )
                    batch_results[idx] = result
                except SystemicProviderError as e:
                    # run_single already sets e.duration_ms and e.tokens
                    batch_results[idx] = EvalRunResult(
                        run_index=run_index,
                        success=False,
                        error=str(e),
                        duration_ms=e.duration_ms or 0.0,
                        tokens=e.tokens or 0,
                        model_tag=model_tag,
                        messages=None,
                        row_index=row_index,
                    )
                    if systemic_error is None:
                        systemic_error = str(e)

        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(_run_one(idx, row, row_index, run_index, model_tag))
            for idx, (row, row_index, run_index, model_tag) in enumerate(work_items)
        ]

        try:
            for task in asyncio.as_completed(tasks):
                await task
        except Exception:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return batch_results, systemic_error

    def _compute_summaries(
        self,
        row_results: list[EvalRowResult],
        n_runs: int,
        pass_threshold: float,
    ) -> dict[str, EvalEvalSummary]:
        """Compute per-eval-function summary statistics.

        Delegates to ``build_summary`` so that cache and runner share the
        exact same aggregation logic.  Failed runs and missing eval scores
        are treated as 0.0 so reliability issues are reflected in evaluation
        quality metrics.
        """
        from osmosis_ai.eval.evaluation.cache import build_summary

        # Convert dataclass rows -> flat list[dict] expected by build_summary
        runs: list[dict[str, Any]] = []
        for row in row_results:
            for run in row.runs:
                runs.append(
                    {
                        "row_index": row.row_index,
                        "model_tag": run.model_tag,
                        "scores": run.scores,
                        "tokens": run.tokens,
                        "duration_ms": run.duration_ms,
                    }
                )

        eval_fn_names = [fn.name for fn in self.eval_fns]
        raw = build_summary(runs, eval_fn_names, pass_threshold, n_runs)

        # Map dict -> EvalEvalSummary
        summaries: dict[str, EvalEvalSummary] = {}
        for name, stats in raw["eval_fns"].items():
            summaries[name] = EvalEvalSummary(
                mean=stats.get("mean", 0.0),
                median=stats.get("median", 0.0),
                std=stats.get("std", 0.0),
                min=stats.get("min", 0.0),
                max=stats.get("max", 0.0),
                p25=stats.get("p25", 0.0),
                p75=stats.get("p75", 0.0),
                pass_at_k=stats.get("pass_at_k", {}),
            )
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
                model_name = self.primary_model_name
            elif tag == "baseline":
                model_name = self.baseline_model_name
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
