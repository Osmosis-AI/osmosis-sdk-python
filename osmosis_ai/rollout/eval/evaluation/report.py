"""Evaluation reporting: pass@k estimator and console output formatting."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout.eval.common.cli import format_duration

if TYPE_CHECKING:
    from osmosis_ai.rollout.console import Console
    from osmosis_ai.rollout.eval.evaluation.runner import EvalResult


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


def _build_model_data(result: EvalResult, model: str) -> list[dict[str, Any]]:
    """Build per-model data list for unified table rendering.

    In comparison mode, returns one entry per model from model_summaries.
    In single-model mode, returns one entry with overall stats.
    """
    if result.model_summaries:
        model_data = []
        for ms in result.model_summaries:
            avg_latency = (
                ms.total_duration_ms / ms.total_runs if ms.total_runs > 0 else 0
            )
            avg_tokens = ms.total_tokens / ms.total_runs if ms.total_runs > 0 else 0
            model_data.append(
                {
                    "name": ms.model or ms.model_tag,
                    "eval_summaries": ms.eval_summaries,
                    "avg_latency_ms": avg_latency,
                    "avg_tokens": avg_tokens,
                }
            )
        return model_data

    avg_latency = (
        result.total_duration_ms / result.total_runs if result.total_runs > 0 else 0
    )
    avg_tokens = result.total_tokens / result.total_runs if result.total_runs > 0 else 0
    return [
        {
            "name": model,
            "eval_summaries": result.eval_summaries,
            "avg_latency_ms": avg_latency,
            "avg_tokens": avg_tokens,
        }
    ]


def format_eval_report(result: EvalResult, console: Console, model: str = "") -> None:
    """Print a formatted evaluation report to the console.

    Renders a unified table with per-model rows for each eval function.

    Args:
        result: The evaluation result to report.
        console: Console instance for output.
        model: Model name for single-model mode display.
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

    if not result.eval_summaries:
        console.print()
        console.print("  No eval results.")
        return

    model_data = _build_model_data(result, model)
    console.print()

    eval_fn_names = list(result.eval_summaries.keys())

    # Collect pass@k columns from all models' summaries
    pass_k_values: list[int] = []
    if result.n_runs > 1:
        for md in model_data:
            for k in _collect_pass_k_values(md["eval_summaries"]):
                if k not in pass_k_values:
                    pass_k_values.append(k)
        pass_k_values.sort()

    if console.run_rich(
        lambda rich_console: _format_tables_rich(
            eval_fn_names,
            model_data,
            pass_k_values,
            rich_console,
        )
    ):
        pass
    else:
        _format_tables_plain(
            eval_fn_names,
            model_data,
            pass_k_values,
            console,
        )

    console.print()
    console.print()


def _format_tables_rich(
    eval_fn_names: list[str],
    model_data: list[dict[str, Any]],
    pass_k_values: list[int],
    rich_console: Any,
) -> None:
    """Render the performance and eval results tables using rich."""
    from rich import box
    from rich.table import Table

    # --- Performance table ---
    perf_table = Table(
        title="Performance",
        box=box.ROUNDED,
        title_style="bold",
        header_style="bold cyan",
        padding=(0, 1),
    )
    perf_table.add_column("Model", style="bold", no_wrap=True)
    perf_table.add_column("Avg Latency", justify="right")
    perf_table.add_column("Avg Tokens", justify="right")
    for md in model_data:
        perf_table.add_row(
            md["name"],
            format_duration(md["avg_latency_ms"]),
            f"{md['avg_tokens']:,.0f}",
        )
    rich_console.print(perf_table)
    rich_console.print()

    # --- Eval results table ---
    table = Table(
        title="Eval Results",
        box=box.ROUNDED,
        title_style="bold",
        header_style="bold cyan",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("Eval Function", style="bold", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Std", justify="right", style="dim")
    for k in pass_k_values:
        table.add_column(f"pass@{k}", justify="right")

    for _fn_idx, fn_name in enumerate(eval_fn_names):
        for _m_idx, md in enumerate(model_data):
            summary = md["eval_summaries"].get(fn_name)
            if summary is None:
                continue

            display_fn = fn_name

            mean_color = _score_color(summary.mean)
            row: list[str] = [
                display_fn,
                md["name"],
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


def _format_tables_plain(
    eval_fn_names: list[str],
    model_data: list[dict[str, Any]],
    pass_k_values: list[int],
    console: Console,
) -> None:
    """Render the performance and eval results tables as plain text."""
    # --- Performance table ---
    perf_header_parts = [
        f"{'Model':<15}",
        f"{'Avg Latency':>12}",
        f"{'Avg Tokens':>11}",
    ]
    perf_header = " | ".join(perf_header_parts)
    console.print("Performance:", style="bold")
    console.print(perf_header, style="bold")
    console.print("-" * len(perf_header))
    for md in model_data:
        perf_row = " | ".join(
            [
                f"{md['name']:<15}",
                f"{format_duration(md['avg_latency_ms']):>12}",
                f"{md['avg_tokens']:>11,.0f}",
            ]
        )
        console.print(perf_row)
    console.print()

    # --- Eval results table ---
    header_parts = [
        f"{'Eval Function':<20}",
        f"{'Model':<15}",
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

    for fn_idx, fn_name in enumerate(eval_fn_names):
        for m_idx, md in enumerate(model_data):
            summary = md["eval_summaries"].get(fn_name)
            if summary is None:
                continue

            display_fn = fn_name

            row_parts = [
                f"{display_fn:<20}",
                f"{md['name']:<15}",
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

            # Separator after every row except the very last
            is_last_model = m_idx == len(model_data) - 1
            is_last_fn = fn_idx == len(eval_fn_names) - 1
            if not (is_last_model and is_last_fn):
                console.print("-" * len(header))


__all__ = [
    "format_eval_report",
    "pass_at_k",
]
