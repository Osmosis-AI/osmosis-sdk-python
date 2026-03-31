"""Render training metric trends as compact sparklines for rich terminals."""

from __future__ import annotations

import math

from osmosis_ai.platform.api.models import MetricDataPoint, MetricHistory

MIN_TREND_TERMINAL_WIDTH = 100
MIN_SPARKLINE_WIDTH = 8
# Two gaps: title/spark and spark/summary (each "  ").
_SEP_BETWEEN_COLUMNS = 2 + 2
SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"


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
    # Huge magnitudes, ultra-tiny, or sub-milli where fixed .3f collapses (e.g. 1e-6 → "0.000").
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


def _title_cell(title: str, width: int) -> str:
    """Fit title to column width (truncate with ellipsis if needed)."""
    if width <= 0:
        return ""
    if len(title) <= width:
        return title.ljust(width)
    if width == 1:
        return "…"
    return title[: width - 1] + "…"


def _values_ordered_by_step(m: MetricHistory) -> list[float]:
    """Values in ascending `MetricDataPoint.step` order (not list order)."""
    ordered: list[MetricDataPoint] = sorted(
        m.data_points, key=lambda dp: (dp.step, dp.timestamp)
    )
    return [dp.value for dp in ordered]


def _summary_pair(m: MetricHistory) -> tuple[str, str]:
    raw = _values_ordered_by_step(m)
    lo = _format_metric_value(raw[0])
    hi = _format_metric_value(raw[-1])
    return lo, hi


def _layout_columns(
    metrics: list[MetricHistory], terminal_width: int
) -> tuple[int, int, int]:
    """Return (title_col_width, summary_col_width, sparkline_width).

    Row layout: ``title | sparkline | summary`` (summary right-aligned in the last column).
    """
    max_title = max(len(m.title) for m in metrics)
    max_summary_len = 0
    for m in metrics:
        if m.data_points:
            lo, hi = _summary_pair(m)
            max_summary_len = max(max_summary_len, len(f"{lo} -> {hi}"))

    if max_summary_len == 0:
        # No series has points: only "title  No data" rows.
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
    # If summary is pathological, re-clamp title so the row still fits.
    if title_col + max_summary_len + _SEP_BETWEEN_COLUMNS + spark_w > terminal_width:
        title_col = max(
            1, terminal_width - max_summary_len - _SEP_BETWEEN_COLUMNS - spark_w
        )
        title_col = min(title_col, max_title)
    return title_col, max_summary_len, spark_w


def _sparkline_for_values(values: list[float]) -> str:
    """Normalize values independently to block characters (flat → mid block)."""
    if not values:
        return ""
    nblocks = len(SPARKLINE_BLOCKS)
    mid = SPARKLINE_BLOCKS[nblocks // 2]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return mid * len(values)
    vmin = min(finite)
    vmax = max(finite)
    if vmin == vmax:
        return mid * len(values)
    out: list[str] = []
    span = vmax - vmin
    # Huge opposite-signed magnitudes can make span overflow to inf → NaN in t.
    if not math.isfinite(span) or span <= 0:
        return mid * len(values)
    for v in values:
        if not math.isfinite(v):
            out.append(mid)
            continue
        t = (v - vmin) / span
        if not math.isfinite(t):
            out.append(mid)
            continue
        idx = round(t * (nblocks - 1))
        idx = max(0, min(nblocks - 1, idx))
        out.append(SPARKLINE_BLOCKS[idx])
    return "".join(out)


def render_metric_trends(
    metrics: list[MetricHistory],
    *,
    terminal_width: int = 120,
) -> str | None:
    """Render ordered metric trends as plain text, or None when there is nothing to show."""
    if not metrics:
        return None

    title_col, summary_col, spark_w = _layout_columns(metrics, terminal_width)
    lines: list[str] = []

    if summary_col == 0:
        for m in metrics:
            lines.append(_title_cell(m.title, title_col) + "  No data")
        return "\n".join(lines)

    for m in metrics:
        if not m.data_points:
            lines.append(_title_cell(m.title, title_col) + "  No data")
            continue
        raw = _values_ordered_by_step(m)
        sampled = _downsample_values(raw, spark_w)
        spark = _sparkline_for_values(sampled)
        spark_cell = spark.ljust(spark_w)[:spark_w]
        lo, hi = _summary_pair(m)
        summ = f"{lo} -> {hi}"
        summ_cell = summ.rjust(summary_col)
        lines.append(
            _title_cell(m.title, title_col) + "  " + spark_cell + "  " + summ_cell
        )
    return "\n".join(lines)
