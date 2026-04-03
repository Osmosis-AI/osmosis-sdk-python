"""Tests for eval runner (v2 — WorkflowExecutor-based)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osmosis_ai.eval.common.dataset import DatasetRow
from osmosis_ai.eval.evaluation.eval_fn import EvalFnWrapper
from osmosis_ai.eval.evaluation.runner import (
    EvalRowResult,
    EvalRunner,
    EvalRunResult,
    _extract_systemic_error_metrics,
)
from osmosis_ai.eval.executor import ExecutionResult, WorkflowExecutor
from osmosis_ai.eval.proxy import RequestMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(index: int = 0) -> DatasetRow:
    return {  # type: ignore[return-value]
        "system_prompt": "Be helpful",
        "user_prompt": f"Question {index}",
        "ground_truth": f"answer {index}",
    }


def _make_eval_fn(name: str = "score", return_value: float = 1.0) -> EvalFnWrapper:
    """Create a mock eval function wrapper."""

    async def fake_fn(
        messages: list[dict[str, Any]],
        ground_truth: str,
        metadata: dict[str, Any],
    ) -> float:
        return return_value

    return EvalFnWrapper(fn=fake_fn, name=name)


def _make_mock_executor(
    success: bool = True,
    messages: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> AsyncMock:
    """Create a mock WorkflowExecutor."""
    mock_executor = AsyncMock(spec=WorkflowExecutor)
    mock_sample = MagicMock()
    mock_sample.messages = messages or [{"role": "assistant", "content": "hello"}]
    mock_executor.run_single.return_value = ExecutionResult(
        success=success,
        samples={"s1": mock_sample} if success else {},
        duration_ms=100.0,
        metrics=RequestMetrics(prompt_tokens=10, completion_tokens=5, num_calls=1),
        error=error if not success else None,
    )
    return mock_executor


# ---------------------------------------------------------------------------
# Tests: run_single
# ---------------------------------------------------------------------------


class TestRunSingle:
    async def test_run_single_success(self) -> None:
        """Success case: workflow succeeds, eval fns produce scores."""
        executor = _make_mock_executor(success=True)
        eval_fns = [_make_eval_fn("score_a", 0.9), _make_eval_fn("score_b", 0.5)]
        runner = EvalRunner(executor=executor, eval_fns=eval_fns)

        result = await runner.run_single(row=_make_row(0), row_index=0, run_index=0)

        assert result.success is True
        assert result.scores["score_a"] == 0.9
        assert result.scores["score_b"] == 0.5
        assert result.tokens == 15  # 10 + 5
        assert result.duration_ms == 100.0
        assert result.messages == [{"role": "assistant", "content": "hello"}]
        assert result.row_index == 0
        assert result.run_index == 0
        assert result.model_tag is None

    async def test_run_single_failure(self) -> None:
        """Workflow failure returns EvalRunResult with success=False."""
        executor = _make_mock_executor(success=False, error="workflow failed")
        runner = EvalRunner(executor=executor, eval_fns=[_make_eval_fn("score")])

        result = await runner.run_single(row=_make_row(0), row_index=0, run_index=0)

        assert result.success is False
        assert result.error == "workflow failed"
        assert result.scores == {}
        assert result.messages is None
        assert result.tokens == 15

    async def test_run_single_baseline(self) -> None:
        """model_tag='baseline' routes to the baseline executor."""
        primary_executor = _make_mock_executor(
            success=True,
            messages=[{"role": "assistant", "content": "primary response"}],
        )
        baseline_executor = _make_mock_executor(
            success=True,
            messages=[{"role": "assistant", "content": "baseline response"}],
        )
        runner = EvalRunner(
            executor=primary_executor,
            eval_fns=[_make_eval_fn("score")],
            baseline_executor=baseline_executor,
        )

        # Primary call
        primary_result = await runner.run_single(
            row=_make_row(0), row_index=0, run_index=0, model_tag="primary"
        )
        assert primary_result.model_tag == "primary"
        assert primary_result.success is True
        primary_executor.run_single.assert_awaited()

        # Baseline call
        baseline_result = await runner.run_single(
            row=_make_row(0), row_index=0, run_index=0, model_tag="baseline"
        )
        assert baseline_result.model_tag == "baseline"
        assert baseline_result.success is True
        baseline_executor.run_single.assert_awaited()

    async def test_run_single_eval_fn_error_returns_zero(self) -> None:
        """When an eval function raises, its score defaults to 0.0."""
        executor = _make_mock_executor(success=True)

        async def failing_fn(
            messages: list[dict[str, Any]],
            ground_truth: str,
            metadata: dict[str, Any],
        ) -> float:
            raise ValueError("boom")

        runner = EvalRunner(
            executor=executor,
            eval_fns=[
                _make_eval_fn("good_fn", 0.75),
                EvalFnWrapper(fn=failing_fn, name="bad_fn"),
            ],
        )

        result = await runner.run_single(row=_make_row(0), row_index=0, run_index=0)

        assert result.success is True
        assert result.scores["good_fn"] == 0.75
        assert result.scores["bad_fn"] == 0.0


# ---------------------------------------------------------------------------
# Tests: run_batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    async def test_run_batch_concurrent(self) -> None:
        """Runs 4 items with batch_size=2 and collects all results."""
        executor = _make_mock_executor(success=True)
        runner = EvalRunner(executor=executor, eval_fns=[_make_eval_fn("score")])

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (_make_row(i), i, 0, None) for i in range(4)
        ]

        results, systemic_error = await runner.run_batch(work_items, batch_size=2)

        assert systemic_error is None
        assert len(results) == 4
        assert all(r is not None for r in results)
        assert all(r.success for r in results)  # type: ignore[union-attr]
        # Verify correct row_index assignment
        for i, r in enumerate(results):
            assert r is not None
            assert r.row_index == i

    async def test_run_batch_empty(self) -> None:
        """Empty work_items returns empty list."""
        executor = _make_mock_executor()
        runner = EvalRunner(executor=executor, eval_fns=[])

        results, systemic_error = await runner.run_batch([], batch_size=4)

        assert results == []
        assert systemic_error is None

    async def test_run_batch_systemic_error(self) -> None:
        """SystemicProviderError is caught and reported."""
        from osmosis_ai.eval.common.errors import SystemicProviderError

        executor = _make_mock_executor(success=True)
        # Make the executor raise SystemicProviderError
        executor.run_single.side_effect = SystemicProviderError("Auth failed")

        runner = EvalRunner(executor=executor, eval_fns=[_make_eval_fn("score")])

        work_items: list[tuple[DatasetRow, int, int, str | None]] = [
            (_make_row(0), 0, 0, None),
            (_make_row(1), 1, 0, None),
        ]

        results, systemic_error = await runner.run_batch(work_items, batch_size=2)

        assert systemic_error is not None
        assert "Auth failed" in systemic_error
        # All items should have results (error results)
        assert all(r is not None for r in results)
        assert all(not r.success for r in results)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests: _extract_messages
# ---------------------------------------------------------------------------


class TestExtractMessages:
    def test_extract_from_sample_with_messages_attr(self) -> None:
        """Extracts messages from a sample object with .messages attribute."""
        sample = MagicMock()
        sample.messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hi"},
        ]
        result = EvalRunner._extract_messages({"s1": sample})
        assert len(result) == 2
        assert result[1]["content"] == "hi"

    def test_extract_from_dict_sample(self) -> None:
        """Extracts messages from a dict sample."""
        samples = {"s1": {"messages": [{"role": "assistant", "content": "dict msg"}]}}
        result = EvalRunner._extract_messages(samples)
        assert len(result) == 1
        assert result[0]["content"] == "dict msg"

    def test_extract_empty_samples(self) -> None:
        """Empty samples returns empty list."""
        result = EvalRunner._extract_messages({})
        assert result == []

    def test_extract_sample_no_messages(self) -> None:
        """Sample with no messages attribute and not a dict returns empty."""
        sample = MagicMock(spec=[])  # no attributes
        result = EvalRunner._extract_messages({"s1": sample})
        assert result == []


# ---------------------------------------------------------------------------
# Tests: has_baseline property
# ---------------------------------------------------------------------------


class TestHasBaseline:
    def test_has_baseline_true(self) -> None:
        """has_baseline returns True when baseline_executor is set."""
        executor = _make_mock_executor()
        baseline = _make_mock_executor()
        runner = EvalRunner(
            executor=executor,
            eval_fns=[],
            baseline_executor=baseline,
        )
        assert runner.has_baseline is True

    def test_has_baseline_false(self) -> None:
        """has_baseline returns False when no baseline_executor."""
        executor = _make_mock_executor()
        runner = EvalRunner(executor=executor, eval_fns=[])
        assert runner.has_baseline is False


# ---------------------------------------------------------------------------
# Tests: _filter_runs_by_tag
# ---------------------------------------------------------------------------


class TestFilterRunsByTag:
    def test_filters_correctly(self) -> None:
        executor = _make_mock_executor()
        runner = EvalRunner(executor=executor, eval_fns=[])

        row_results = [
            EvalRowResult(
                row_index=0,
                runs=[
                    EvalRunResult(run_index=0, success=True, model_tag="primary"),
                    EvalRunResult(run_index=0, success=True, model_tag="baseline"),
                ],
            ),
            EvalRowResult(
                row_index=1,
                runs=[
                    EvalRunResult(run_index=0, success=True, model_tag="primary"),
                    EvalRunResult(run_index=0, success=True, model_tag="baseline"),
                ],
            ),
        ]

        primary_only = runner._filter_runs_by_tag(row_results, "primary")
        assert len(primary_only) == 2
        for row in primary_only:
            assert all(r.model_tag == "primary" for r in row.runs)

    def test_no_matching_tag_returns_empty(self) -> None:
        executor = _make_mock_executor()
        runner = EvalRunner(executor=executor, eval_fns=[])

        row_results = [
            EvalRowResult(
                row_index=0,
                runs=[
                    EvalRunResult(run_index=0, success=True, model_tag="primary"),
                ],
            ),
        ]

        result = runner._filter_runs_by_tag(row_results, "baseline")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _extract_systemic_error_metrics
# ---------------------------------------------------------------------------


class TestExtractSystemicErrorMetrics:
    def test_uses_attributes_when_present(self) -> None:
        from osmosis_ai.eval.common.errors import SystemicProviderError

        e = SystemicProviderError("test")
        e.duration_ms = 1234.5
        e.tokens = 42
        dur, tok = _extract_systemic_error_metrics(e, fallback_started_at=0.0)
        assert dur == 1234.5
        assert tok == 42

    def test_fallback_duration_when_missing(self) -> None:
        from osmosis_ai.eval.common.errors import SystemicProviderError

        e = SystemicProviderError("test")
        started = time.monotonic() - 0.5  # 500ms ago
        dur, tok = _extract_systemic_error_metrics(e, fallback_started_at=started)
        assert dur >= 400.0
        assert tok == 0

    def test_fallback_tokens_when_none(self) -> None:
        from osmosis_ai.eval.common.errors import SystemicProviderError

        e = SystemicProviderError("test")
        e.duration_ms = 100.0
        e.tokens = None
        dur, tok = _extract_systemic_error_metrics(e, fallback_started_at=0.0)
        assert dur == 100.0
        assert tok == 0


# ---------------------------------------------------------------------------
# Tests: _compute_summaries
# ---------------------------------------------------------------------------


class TestComputeSummaries:
    def test_empty_rows_produces_empty_summaries(self) -> None:
        executor = _make_mock_executor()
        runner = EvalRunner(
            executor=executor,
            eval_fns=[_make_eval_fn("score")],
        )

        summaries = runner._compute_summaries([], n_runs=1, pass_threshold=1.0)
        assert "score" in summaries
        summary = summaries["score"]
        assert summary.mean == 0.0
        assert summary.std == 0.0

    def test_missing_score_defaults_to_zero(self) -> None:
        executor = _make_mock_executor()
        runner = EvalRunner(
            executor=executor,
            eval_fns=[_make_eval_fn("score")],
        )

        row_results = [
            EvalRowResult(
                row_index=0,
                runs=[
                    EvalRunResult(
                        run_index=0,
                        success=True,
                        scores={"score": 1.0},
                        tokens=10,
                    ),
                    EvalRunResult(
                        run_index=1,
                        success=False,
                        scores={},
                        tokens=0,
                    ),
                ],
            ),
        ]

        summaries = runner._compute_summaries(row_results, n_runs=2, pass_threshold=0.5)
        summary = summaries["score"]
        assert summary.mean == pytest.approx(0.5)
        assert summary.min == 0.0
        assert summary.max == 1.0


# ---------------------------------------------------------------------------
# Tests: _compute_model_summaries
# ---------------------------------------------------------------------------


class TestComputeModelSummaries:
    def test_model_names_used(self) -> None:
        executor = _make_mock_executor()
        runner = EvalRunner(
            executor=executor,
            eval_fns=[_make_eval_fn("score")],
            baseline_executor=_make_mock_executor(),
            primary_model_name="gpt-4o",
            baseline_model_name="gpt-3.5",
        )

        row_results = [
            EvalRowResult(
                row_index=0,
                runs=[
                    EvalRunResult(
                        run_index=0,
                        success=True,
                        scores={"score": 1.0},
                        model_tag="primary",
                    ),
                    EvalRunResult(
                        run_index=0,
                        success=True,
                        scores={"score": 0.5},
                        model_tag="baseline",
                    ),
                ],
            ),
        ]

        model_summaries = runner._compute_model_summaries(
            row_results, n_runs=1, pass_threshold=0.5
        )

        assert len(model_summaries) == 2
        assert model_summaries[0].model == "gpt-4o"
        assert model_summaries[0].model_tag == "primary"
        assert model_summaries[1].model == "gpt-3.5"
        assert model_summaries[1].model_tag == "baseline"
