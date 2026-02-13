"""Evaluation reporting: pass@k estimator and console output formatting."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout.eval.common.cli import format_duration

if TYPE_CHECKING:
    from osmosis_ai.rollout.eval.evaluation.runner import EvalResult
    from osmosis_ai.rollout.console import Console


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Computes the probability that at least one of k randomly selected
    samples from n total samples (of which c are correct) is correct.

    Formula: 1 - comb(n-c, k) / comb(n, k)

    Args:
        n: Total number of samples.
        c: Number of correct (passing) samples.
        k: Number of samples to select.

    Returns:
        Estimated pass@k probability in [0, 1].
    """
    if c == 0:
        return 0.0
    if n <= k:
        return 1.0
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _format_score(value: float) -> str:
    """Format a score value, showing integers without decimals."""
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def _format_percent(value: float) -> str:
    """Format a 0-1 probability as a percentage string."""
    return f"{value * 100:.1f}%"


def _score_color(value: float) -> str:
    """Return a rich color name based on score value (0-1 scale)."""
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


def _collect_pass_k_values(eval_summaries: dict[str, Any]) -> list[int]:
    """Collect sorted pass@k column values from eval summaries."""
    pass_k_values: list[int] = []
    for summary in eval_summaries.values():
        for k in summary.pass_at_k:
            if k not in pass_k_values:
                pass_k_values.append(k)
    pass_k_values.sort()
    return pass_k_values


def format_eval_report(result: "EvalResult", console: "Console") -> None:
    """Print a formatted evaluation report to the console.

    Displays a table of eval function statistics including mean, min, max,
    std, and pass@k columns (when n > 1).  When model_summaries are present
    (comparison mode), a per-model breakdown and win/loss/tie table is shown.

    Args:
        result: The evaluation result to report.
        console: Console instance for output.
    """
    console.print()
    console.print("Evaluation Results:", style="bold")
    console.print(f"  Rows: {result.total_rows}")
    console.print(f"  Runs per row: {result.n_runs}")
    console.print(f"  Total runs: {result.total_runs}")
    console.print(f"  Duration: {format_duration(result.total_duration_ms)}")
    console.print(f"  Total tokens: {result.total_tokens:,}")
    if result.stopped_early:
        reason = f" Reason: {result.stop_reason}" if result.stop_reason else ""
        console.print(
            f"  Stopped early after a failed run.{reason}",
            style="yellow",
        )
    console.print()

    if not result.eval_summaries:
        console.print("  No eval results.")
        return

    # Determine pass@k columns
    pass_k_values = (
        _collect_pass_k_values(result.eval_summaries)
        if result.n_runs > 1
        else []
    )

    # Standard single-model report
    if console.run_rich(
        lambda rich_console: _format_eval_report_rich(
            result,
            pass_k_values,
            rich_console,
        )
    ):
        pass
    else:
        _format_eval_report_plain(result, console, pass_k_values)

    # Comparison report (if baseline was used)
    if result.comparisons:
        if console.run_rich(
            lambda rich_console: _format_comparison_report_rich(
                result,
                rich_console,
            )
        ):
            return
        _format_comparison_report_plain(result, console)


def _format_eval_report_rich(
    result: "EvalResult",
    pass_k_values: list[int],
    rich_console: Any,
) -> None:
    """Render the evaluation table using rich."""
    from rich.table import Table
    from rich import box

    table = Table(
        title="Eval Scores",
        box=box.ROUNDED,
        title_style="bold",
        header_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Eval Function", style="bold", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Std", justify="right", style="dim")
    for k in pass_k_values:
        table.add_column(f"pass@{k}", justify="right")

    for name, summary in result.eval_summaries.items():
        mean_color = _score_color(summary.mean)
        row: list[str] = [
            name,
            f"[bold {mean_color}]{_format_score(summary.mean)}[/bold {mean_color}]",
            _format_score(summary.min),
            _format_score(summary.max),
            _format_score(summary.std),
        ]
        for k in pass_k_values:
            val = summary.pass_at_k.get(k)
            if val is not None:
                color = _score_color(val)
                row.append(f"[{color}]{_format_percent(val)}[/{color}]")
            else:
                row.append("[dim]N/A[/dim]")
        table.add_row(*row)

    rich_console.print(table)


def _format_eval_report_plain(
    result: "EvalResult",
    console: "Console",
    pass_k_values: list[int],
) -> None:
    """Render the evaluation table as plain text."""
    # Build header
    header_parts = [
        f"{'Eval Function':<20}",
        f"{'Mean':>6}",
        f"{'Min':>6}",
        f"{'Max':>6}",
        f"{'Std':>6}",
    ]
    for k in pass_k_values:
        header_parts.append(f"{'pass@' + str(k):>8}")
    header = " | ".join(header_parts)

    console.print(header, style="bold")
    console.print("-" * len(header))

    for name, summary in result.eval_summaries.items():
        row_parts = [
            f"{name:<20}",
            f"{summary.mean:>6.3f}",
            f"{summary.min:>6.3f}",
            f"{summary.max:>6.3f}",
            f"{summary.std:>6.3f}",
        ]
        for k in pass_k_values:
            val = summary.pass_at_k.get(k)
            if val is not None:
                row_parts.append(f"{val * 100:>7.1f}%")
            else:
                row_parts.append(f"{'N/A':>8}")
        console.print(" | ".join(row_parts))

def _delta_color(delta: float) -> str:
    """Return a color name based on delta sign."""
    if delta > 0:
        return "green"
    if delta < 0:
        return "red"
    return "dim"


def _format_comparison_report_rich(
    result: "EvalResult",
    rich_console: Any,
) -> None:
    """Render the model comparison table using rich."""
    from rich.table import Table
    from rich import box

    if not result.comparisons or not result.model_summaries:
        return

    primary_label = "primary"
    baseline_label = "baseline"
    for ms in result.model_summaries:
        if ms.model_tag == "primary":
            primary_label = ms.model or "primary"
        elif ms.model_tag == "baseline":
            baseline_label = ms.model or "baseline"

    table = Table(
        title="Model Comparison",
        box=box.ROUNDED,
        title_style="bold",
        header_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Eval Function", style="bold", no_wrap=True)
    table.add_column(f"Primary ({primary_label})", justify="right")
    table.add_column(f"Baseline ({baseline_label})", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Tie", justify="right")

    for comp in result.comparisons:
        d_color = _delta_color(comp.delta)
        delta_sign = "+" if comp.delta > 0 else ""
        table.add_row(
            comp.eval_fn,
            f"[{_score_color(comp.primary_mean)}]{_format_score(comp.primary_mean)}[/{_score_color(comp.primary_mean)}]",
            f"[{_score_color(comp.baseline_mean)}]{_format_score(comp.baseline_mean)}[/{_score_color(comp.baseline_mean)}]",
            f"[{d_color}]{delta_sign}{_format_score(comp.delta)}[/{d_color}]",
            str(comp.wins),
            str(comp.losses),
            str(comp.ties),
        )

    rich_console.print()
    rich_console.print(table)


def _format_comparison_report_plain(
    result: "EvalResult",
    console: "Console",
) -> None:
    """Render the model comparison table as plain text."""
    if not result.comparisons or not result.model_summaries:
        return

    console.print()
    console.print("Model Comparison:", style="bold")

    header_parts = [
        f"{'Eval Function':<20}",
        f"{'Primary':>8}",
        f"{'Baseline':>8}",
        f"{'Delta':>8}",
        f"{'Win':>5}",
        f"{'Loss':>5}",
        f"{'Tie':>5}",
    ]
    header = " | ".join(header_parts)

    console.print(header, style="bold")
    console.print("-" * len(header))

    for comp in result.comparisons:
        delta_sign = "+" if comp.delta > 0 else ""
        row_parts = [
            f"{comp.eval_fn:<20}",
            f"{comp.primary_mean:>8.3f}",
            f"{comp.baseline_mean:>8.3f}",
            f"{delta_sign}{comp.delta:>7.3f}",
            f"{comp.wins:>5}",
            f"{comp.losses:>5}",
            f"{comp.ties:>5}",
        ]
        console.print(" | ".join(row_parts))


__all__ = [
    "format_eval_report",
    "pass_at_k",
]
