"""Benchmark reporting: pass@k estimator and console output formatting."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

from osmosis_ai.rollout.eval.common.cli import format_duration

if TYPE_CHECKING:
    from osmosis_ai.rollout.eval.bench.runner import BenchResult
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
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
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


def format_bench_report(result: "BenchResult", console: "Console") -> None:
    """Print a formatted benchmark report to the console.

    Displays a table of eval function statistics including mean, min, max,
    std, and pass@k columns (when n > 1).

    Args:
        result: The benchmark result to report.
        console: Console instance for output.
    """
    console.print()
    console.print("Benchmark Results:", style="bold")
    console.print(f"  Rows: {result.total_rows}")
    console.print(f"  Runs per row: {result.n_runs}")
    console.print(f"  Total runs: {result.total_runs}")
    console.print(f"  Duration: {format_duration(result.total_duration_ms)}")
    console.print(f"  Total tokens: {result.total_tokens:,}")
    console.print()

    if not result.eval_summaries:
        console.print("  No eval results.")
        return

    # Determine pass@k columns
    pass_k_values: list[int] = []
    if result.n_runs > 1:
        for summary in result.eval_summaries.values():
            for k in summary.pass_at_k:
                if k not in pass_k_values:
                    pass_k_values.append(k)
        pass_k_values.sort()

    if console.use_rich:
        _format_bench_report_rich(result, console, pass_k_values)
    else:
        _format_bench_report_plain(result, console, pass_k_values)


def _format_bench_report_rich(
    result: "BenchResult",
    console: "Console",
    pass_k_values: list[int],
) -> None:
    """Render the benchmark table using rich."""
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

    console._rich.print(table)  # type: ignore[union-attr]


def _format_bench_report_plain(
    result: "BenchResult",
    console: "Console",
    pass_k_values: list[int],
) -> None:
    """Render the benchmark table as plain text."""
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

__all__ = [
    "format_bench_report",
    "pass_at_k",
]
