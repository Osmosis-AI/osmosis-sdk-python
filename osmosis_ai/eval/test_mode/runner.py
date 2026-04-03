"""Test mode runner — thin wrapper over WorkflowExecutor."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from osmosis_ai.eval.common.dataset import DatasetRow, dataset_row_to_prompt
from osmosis_ai.eval.executor import WorkflowExecutor


@dataclass
class TestRunResult:
    """Result from running a single dataset row in test mode."""

    row_index: int
    success: bool
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    samples: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestBatchResult:
    """Aggregated results from running a batch of rows."""

    results: list[TestRunResult]
    total: int
    passed: int
    failed: int
    total_duration_ms: float
    total_tokens: int
    stopped_early: bool = False
    stop_reason: str | None = None


class TestRunner:
    """Test mode runner using WorkflowExecutor."""

    def __init__(self, executor: WorkflowExecutor) -> None:
        self.executor = executor

    async def run_single(self, row: DatasetRow, row_index: int) -> TestRunResult:
        prompt = dataset_row_to_prompt(row)
        rollout_id = f"test-{row_index}"
        result = await self.executor.run_single(prompt, rollout_id)
        return TestRunResult(
            row_index=row_index,
            success=result.success,
            error=result.error,
            duration_ms=result.duration_ms,
            token_usage={
                "prompt_tokens": result.metrics.prompt_tokens,
                "completion_tokens": result.metrics.completion_tokens,
                "total_tokens": result.metrics.total_tokens,
                "num_llm_calls": result.metrics.num_calls,
            },
            samples=result.samples,
        )

    async def run_batch(
        self,
        rows: list[DatasetRow],
        on_progress: Callable[[int, int, TestRunResult], None] | None = None,
        start_index: int = 0,
    ) -> TestBatchResult:
        results: list[TestRunResult] = []
        total_start = time.monotonic()
        stopped_early = False
        stop_reason: str | None = None
        consecutive_failures = 0

        for i, row in enumerate(rows):
            row_index = start_index + i
            result = await self.run_single(row, row_index)
            results.append(result)
            if on_progress:
                on_progress(i + 1, len(rows), result)

            # Fail fast: proxy-level systemic error (auth, model not found, etc.)
            systemic = self.executor.proxy.systemic_error
            if systemic:
                stopped_early = True
                stop_reason = systemic
                break

            # Fail fast: consecutive failures with 0 tokens means nothing is working
            if not result.success and result.token_usage.get("total_tokens", 0) == 0:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    stopped_early = True
                    stop_reason = result.error
                    break
            else:
                consecutive_failures = 0

        passed = sum(1 for r in results if r.success)
        total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in results)

        return TestBatchResult(
            results=results,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            total_duration_ms=(time.monotonic() - total_start) * 1000,
            total_tokens=total_tokens,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


__all__ = ["TestBatchResult", "TestRunResult", "TestRunner"]
