"""Tests for evaluation report formatting."""

from __future__ import annotations

from io import StringIO

import osmosis_ai.eval.evaluation.report as report_module
from osmosis_ai.cli.console import Console
from osmosis_ai.eval.evaluation.runner import EvalEvalSummary, EvalResult


def _make_eval_result() -> EvalResult:
    """Create a minimal eval result fixture for report formatting tests."""
    return EvalResult(
        rows=[],
        eval_summaries={
            "accuracy": EvalEvalSummary(
                mean=0.75,
                median=0.75,
                std=0.2,
                min=0.5,
                max=1.0,
                p25=0.625,
                p75=0.875,
                pass_at_k={1: 1.0},
            )
        },
        total_rows=1,
        total_runs=2,
        total_tokens=123,
        total_duration_ms=1500.0,
        n_runs=2,
        pass_threshold=0.5,
    )


def test_format_eval_report_renders_results() -> None:
    """Report formatting should render eval results with Rich."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    report_module.format_eval_report(_make_eval_result(), console, model="test-model")

    text = output.getvalue()
    assert "Evaluation Results:" in text
    assert "Eval Function" in text
    assert "accuracy" in text
    assert "Median" in text
    assert "pass@1" in text


def test_format_eval_report_non_tty() -> None:
    """Report formatting should work in non-TTY mode (Rich strips ANSI)."""
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    report_module.format_eval_report(_make_eval_result(), console, model="test-model")

    text = output.getvalue()
    assert "Evaluation Results:" in text
    assert "accuracy" in text
    # Non-TTY output should not contain ANSI escape codes
    assert "\033[" not in text
