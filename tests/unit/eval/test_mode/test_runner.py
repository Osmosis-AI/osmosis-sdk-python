"""Tests for test_mode runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from osmosis_ai.eval.common.dataset import DatasetRow
from osmosis_ai.eval.executor import ExecutionResult, WorkflowExecutor
from osmosis_ai.eval.proxy import RequestMetrics
from osmosis_ai.eval.test_mode.runner import TestRunner, TestRunResult


def _make_row() -> DatasetRow:
    return {"system_prompt": "Be helpful", "user_prompt": "Hi", "ground_truth": "hello"}


async def test_run_single_success():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={},
        duration_ms=100.0,
        metrics=RequestMetrics(prompt_tokens=10, completion_tokens=5, num_calls=1),
    )
    runner = TestRunner(executor=mock_executor)
    result = await runner.run_single(_make_row(), row_index=0)
    assert result.success
    assert result.token_usage["total_tokens"] == 15


async def test_run_single_failure():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.run_single.return_value = ExecutionResult(
        success=False,
        error="crashed",
        duration_ms=50.0,
        metrics=RequestMetrics(),
    )
    runner = TestRunner(executor=mock_executor)
    result = await runner.run_single(_make_row(), row_index=0)
    assert not result.success
    assert result.error == "crashed"


async def test_run_batch():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.proxy = MagicMock(systemic_error=None)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={},
        duration_ms=50.0,
        metrics=RequestMetrics(prompt_tokens=5, completion_tokens=3),
    )
    runner = TestRunner(executor=mock_executor)
    batch = await runner.run_batch([_make_row(), _make_row()], start_index=0)
    assert batch.total == 2
    assert batch.passed == 2
    assert batch.failed == 0
    assert batch.total_tokens == 16


async def test_run_single_row_index_passed_to_rollout_id():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={},
        duration_ms=10.0,
        metrics=RequestMetrics(),
    )
    runner = TestRunner(executor=mock_executor)
    await runner.run_single(_make_row(), row_index=42)
    mock_executor.run_single.assert_called_once()
    _, _kwargs = mock_executor.run_single.call_args
    # positional args: prompt, rollout_id
    args = mock_executor.run_single.call_args.args
    assert args[1] == "test-42"


async def test_run_batch_progress_callback():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.proxy = MagicMock(systemic_error=None)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={},
        duration_ms=10.0,
        metrics=RequestMetrics(prompt_tokens=1, completion_tokens=1),
    )
    runner = TestRunner(executor=mock_executor)
    progress_calls: list[tuple[int, int, TestRunResult]] = []

    def on_progress(current: int, total: int, result: TestRunResult) -> None:
        progress_calls.append((current, total, result))

    await runner.run_batch(
        [_make_row(), _make_row(), _make_row()], on_progress=on_progress
    )
    assert len(progress_calls) == 3
    assert progress_calls[0][0] == 1
    assert progress_calls[0][1] == 3
    assert progress_calls[2][0] == 3


async def test_run_batch_with_start_index():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.proxy = MagicMock(systemic_error=None)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={},
        duration_ms=10.0,
        metrics=RequestMetrics(),
    )
    runner = TestRunner(executor=mock_executor)
    batch = await runner.run_batch([_make_row(), _make_row()], start_index=50)
    assert batch.results[0].row_index == 50
    assert batch.results[1].row_index == 51


async def test_run_batch_with_failures():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.proxy = MagicMock(systemic_error=None)
    mock_executor.run_single.side_effect = [
        ExecutionResult(
            success=True,
            samples={},
            duration_ms=10.0,
            metrics=RequestMetrics(prompt_tokens=2, completion_tokens=2),
        ),
        ExecutionResult(
            success=False,
            error="boom",
            duration_ms=5.0,
            metrics=RequestMetrics(prompt_tokens=1, completion_tokens=0),
        ),
    ]
    runner = TestRunner(executor=mock_executor)
    batch = await runner.run_batch([_make_row(), _make_row()])
    assert batch.total == 2
    assert batch.passed == 1
    assert batch.failed == 1
    assert batch.results[1].error == "boom"


async def test_run_single_maps_all_metrics():
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_executor.run_single.return_value = ExecutionResult(
        success=True,
        samples={"key": "val"},
        duration_ms=200.0,
        metrics=RequestMetrics(prompt_tokens=20, completion_tokens=10, num_calls=3),
    )
    runner = TestRunner(executor=mock_executor)
    result = await runner.run_single(_make_row(), row_index=7)
    assert result.row_index == 7
    assert result.duration_ms == 200.0
    assert result.token_usage["prompt_tokens"] == 20
    assert result.token_usage["completion_tokens"] == 10
    assert result.token_usage["total_tokens"] == 30
    assert result.token_usage["num_llm_calls"] == 3
    assert result.samples == {"key": "val"}
