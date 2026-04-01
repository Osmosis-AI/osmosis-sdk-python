"""Render training metric trends as compact sparklines for rich terminals."""

from __future__ import annotations

import math

from rich.table import Table
from rich.text import Text

from osmosis_ai.platform.api.models import MetricDataPoint, MetricHistory

MIN_TREND_TERMINAL_WIDTH = 100
MIN_SPARKLINE_WIDTH = 8
SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"

# RGB color at each of the 8 block levels (dark-green → bright-green gradient).
_LEVEL_COLORS: list[tuple[int, int, int]] = [
    (11, 74, 28),
    (14, 96, 36),
    (18, 118, 45),
    (23, 140, 54),
    (30, 162, 62),
    (40, 184, 70),
    (55, 206, 78),
    (72, 228, 86),
]


def should_render_metric_trends(
    is_tty: bool,
    terminal_width: int,
    metrics: list[MetricHistory],
) -> bool:
    """Whether trend rendering should run (TTY + width + non-empty metrics)."""
    if not is_tty:
        return False
    if terminal_width < MIN_TREND_TERMINAL_WIDTH:
        return False
    return bool(metrics)


def _format_metric_value(v: float) -> str:
    """Format a metric for display: integers without fraction, else 3 decimals."""
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    av = abs(v)
    # Huge magnitudes, ultra-tiny, or sub-milli where fixed .3f collapses (e.g. 1e-6 -> "0.000").
    if av >= 1e15 or (av > 0 and av < 1e-3):
        s = f"{v:.4g}"
        return s if len(s) <= 18 else f"{v:.3e}"
    r = round(v)
    if abs(v - r) < 1e-9:
        return str(int(r))
    return f"{v:.3f}"


def _downsample_values(values: list[float], target_len: int) -> list[float]:
    """Evenly spaced deterministic downsampling (inclusive endpoints)."""
    if not values:
        return []
    if target_len < 1:
        return []
    if len(values) <= target_len:
        return list(values)
    if target_len == 1:
        return [values[-1]]
    last = len(values) - 1
    return [values[round(i * last / (target_len - 1))] for i in range(target_len)]


def _values_ordered_by_step(m: MetricHistory) -> list[float]:
    """Values in ascending ``MetricDataPoint.step`` order (not list order)."""
    ordered: list[MetricDataPoint] = sorted(
        m.data_points, key=lambda dp: (dp.step, dp.timestamp)
    )
    return [dp.value for dp in ordered]


# Two gaps: title/spark and spark/summary (each "  ").
_SEP_BETWEEN_COLUMNS = 2 + 2


def _layout_columns(
    metrics: list[MetricHistory],
    terminal_width: int,
    summary_pairs: dict[str, tuple[str, str]],
) -> tuple[int, int, int]:
    """Return (title_col_width, summary_col_width, sparkline_width)."""
    max_title = max(len(m.title) for m in metrics)
    max_summary_len = 0
    for m in metrics:
        pair = summary_pairs.get(m.metric_key)
        if pair is not None:
            lo, hi = pair
            max_summary_len = max(max_summary_len, len(lo) + 4 + len(hi))

    if max_summary_len == 0:
        title_col = min(
            max_title, max(1, terminal_width - _SEP_BETWEEN_COLUMNS - len("No data"))
        )
        return title_col, 0, 0

    spark_w = terminal_width - max_title - max_summary_len - _SEP_BETWEEN_COLUMNS
    title_col = max_title
    if spark_w < MIN_SPARKLINE_WIDTH:
        title_col = min(
            max_title,
            max(
                1,
                terminal_width
                - max_summary_len
                - _SEP_BETWEEN_COLUMNS
                - MIN_SPARKLINE_WIDTH,
            ),
        )
        spark_w = terminal_width - title_col - max_summary_len - _SEP_BETWEEN_COLUMNS
    spark_w = max(MIN_SPARKLINE_WIDTH, spark_w)
    if title_col + max_summary_len + _SEP_BETWEEN_COLUMNS + spark_w > terminal_width:
        title_col = max(
            1, terminal_width - max_summary_len - _SEP_BETWEEN_COLUMNS - spark_w
        )
        title_col = min(title_col, max_title)
    return title_col, max_summary_len, spark_w


def _flat_sparkline(count: int) -> Text:
    """A flat mid-level sparkline (used when values have no meaningful range)."""
    nblocks = len(SPARKLINE_BLOCKS)
    mid_idx = nblocks // 2
    ch = SPARKLINE_BLOCKS[mid_idx]
    r, g, b = _LEVEL_COLORS[mid_idx]
    text = Text()
    text.append(ch * count, style=f"rgb({r},{g},{b})")
    return text


def _sparkline_for_values(values: list[float], width: int) -> Text:
    """Normalize values to colored block characters as a Rich Text object."""
    nblocks = len(SPARKLINE_BLOCKS)
    mid_idx = nblocks // 2

    if not values:
        return Text()

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return _flat_sparkline(len(values))

    vmin = min(finite)
    vmax = max(finite)
    span = vmax - vmin
    if vmin == vmax or not math.isfinite(span) or span <= 0:
        return _flat_sparkline(len(values))

    text = Text()
    for v in values:
        if not math.isfinite(v):
            idx = mid_idx
        else:
            t = (v - vmin) / span
            if not math.isfinite(t):
                idx = mid_idx
            else:
                idx = round(t * (nblocks - 1))
                idx = max(0, min(nblocks - 1, idx))
        r, g, b = _LEVEL_COLORS[idx]
        text.append(SPARKLINE_BLOCKS[idx], style=f"rgb({r},{g},{b})")

    # Pad to target width if needed.
    produced = len(values)
    if produced < width:
        text.append(" " * (width - produced))

    return text


def _title_cell(title: str, width: int) -> str:
    """Fit title to column width (truncate with ellipsis if needed)."""
    if width <= 0:
        return ""
    if len(title) <= width:
        return title
    if width == 1:
        return "\u2026"
    return title[: width - 1] + "\u2026"


def render_metric_trends(
    metrics: list[MetricHistory],
    *,
    terminal_width: int = 120,
) -> Table | None:
    """Render ordered metric trends as a Rich Table, or None when nothing to show."""
    if not metrics:
        return None

    # Precompute sorted values and summary pairs once per metric (avoids re-sorting).
    ordered_cache: dict[str, list[float]] = {}
    summary_pairs: dict[str, tuple[str, str]] = {}
    for m in metrics:
        if m.data_points:
            raw = _values_ordered_by_step(m)
            ordered_cache[m.metric_key] = raw
            lo = _format_metric_value(raw[0])
            hi = _format_metric_value(raw[-1])
            summary_pairs[m.metric_key] = (lo, hi)

    title_col, summary_col, spark_w = _layout_columns(
        metrics, terminal_width, summary_pairs
    )

    table = Table(
        box=None,
        show_header=False,
        padding=(0, 1),
        pad_edge=False,
        expand=False,
    )

    if summary_col == 0:
        table.add_column("metric", width=title_col, no_wrap=True, style="bold")
        table.add_column("status", style="dim")
        for m in metrics:
            table.add_row(Text(_title_cell(m.title, title_col)), "No data")
        return table

    table.add_column("metric", width=title_col, no_wrap=True, style="bold")
    table.add_column("trend", width=spark_w, no_wrap=True)
    table.add_column(
        "range", width=summary_col, justify="right", style="dim", no_wrap=True
    )

    for i, m in enumerate(metrics):
        if i > 0:
            table.add_row("", Text(), "")

        if not m.data_points:
            table.add_row(
                Text(_title_cell(m.title, title_col)),
                Text("No data", style="dim"),
                "",
            )
            continue

        raw = ordered_cache[m.metric_key]
        sampled = _downsample_values(raw, spark_w)
        spark = _sparkline_for_values(sampled, spark_w)
        lo, hi = summary_pairs[m.metric_key]
        table.add_row(Text(_title_cell(m.title, title_col)), spark, f"{lo} -> {hi}")

    return table
