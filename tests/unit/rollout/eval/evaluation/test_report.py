"""Tests for evaluation report formatting."""

from __future__ import annotations

from io import StringIO

import osmosis_ai.rollout.eval.evaluation.report as report_module
from osmosis_ai.cli.console import Console
from osmosis_ai.rollout.eval.evaluation.runner import EvalEvalSummary, EvalResult


def _make_eval_result() -> EvalResult:
    """Create a minimal eval result fixture for report formatting tests."""
    return EvalResult(
        rows=[],
        eval_summaries={
            "accuracy": EvalEvalSummary(
                mean=0.75,
                std=0.2,
                min=0.5,
                max=1.0,
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


def test_format_eval_report_falls_back_when_rich_render_fails(monkeypatch) -> None:
    """Report formatting should degrade to plain output when rich rendering fails."""
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    class _FakeRichConsole:
        def print(self, *args: object, end: str = "\n", **kwargs: object) -> None:
            text = " ".join(str(arg) for arg in args)
            output.write(f"{text}{end}")

    console._use_rich = True
    console._rich = _FakeRichConsole()

    def _raise_import_error(*args: object, **kwargs: object) -> None:
        raise ImportError("rich unavailable")

    monkeypatch.setattr(report_module, "_format_tables_rich", _raise_import_error)

    report_module.format_eval_report(_make_eval_result(), console, model="test-model")

    text = output.getvalue()
    assert "Evaluation Results:" in text
    assert "Eval Function" in text
    assert "accuracy" in text
    assert "pass@1" in text
