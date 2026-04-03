"""Tests for WorkflowExecutor and ExecutionResult."""

from __future__ import annotations

from unittest.mock import MagicMock

from osmosis_ai.eval.executor import ExecutionResult, WorkflowExecutor
from osmosis_ai.eval.proxy import EvalProxy, RequestMetrics

# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


def test_execution_result_success():
    r = ExecutionResult(
        success=True,
        samples={"s1": "sample"},
        duration_ms=100.0,
        metrics=RequestMetrics(prompt_tokens=10),
    )
    assert r.success
    assert r.error is None


def test_execution_result_failure():
    r = ExecutionResult(
        success=False,
        error="crashed",
        duration_ms=50.0,
        metrics=RequestMetrics(),
    )
    assert not r.success
    assert r.error == "crashed"
    assert r.samples == {}


# ---------------------------------------------------------------------------
# WorkflowExecutor
# ---------------------------------------------------------------------------


async def test_workflow_executor_success():
    mock_proxy = MagicMock(spec=EvalProxy)
    mock_proxy.url = "http://127.0.0.1:9999"
    mock_proxy.collect_metrics.return_value = RequestMetrics(
        prompt_tokens=10,
        completion_tokens=5,
        num_calls=1,
        total_latency_ms=100.0,
    )

    class FakeWorkflow:
        def __init__(self, config):
            pass

        async def run(self, ctx):
            pass

    executor = WorkflowExecutor(
        workflow_cls=FakeWorkflow,
        workflow_config=None,
        proxy=mock_proxy,
    )
    result = await executor.run_single(
        prompt=[
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ],
        rollout_id="eval-0-run-0",
    )

    assert result.success is True
    assert result.error is None
    assert result.duration_ms > 0
    assert result.metrics.prompt_tokens == 10
    mock_proxy.collect_metrics.assert_called_once_with("eval-0-run-0")


async def test_workflow_executor_handles_exception():
    mock_proxy = MagicMock(spec=EvalProxy)
    mock_proxy.url = "http://127.0.0.1:9999"
    mock_proxy.collect_metrics.return_value = RequestMetrics()

    class CrashingWorkflow:
        def __init__(self, config):
            pass

        async def run(self, ctx):
            raise RuntimeError("tool not found")

    executor = WorkflowExecutor(
        workflow_cls=CrashingWorkflow,
        workflow_config=None,
        proxy=mock_proxy,
    )
    result = await executor.run_single(
        prompt=[{"role": "user", "content": "crash"}],
        rollout_id="eval-1-run-0",
    )

    assert result.success is False
    assert "tool not found" in result.error
    assert result.duration_ms > 0
