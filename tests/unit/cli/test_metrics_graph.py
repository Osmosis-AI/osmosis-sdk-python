"""Tests for osmosis_ai.cli.metrics_graph — terminal metric trend rendering."""

from __future__ import annotations

import re

from rich.console import Console as RichConsole
from rich.table import Table

from osmosis_ai.cli.metrics_graph import (
    MIN_TREND_TERMINAL_WIDTH,
    SPARKLINE_BLOCKS,
    _downsample_values,
    render_metric_trends,
    should_render_metric_trends,
)
from osmosis_ai.platform.api.models import MetricDataPoint, MetricHistory


def _history(
    title: str,
    *,
    key: str = "m/k",
    steps: list[tuple[int, float, int]],
) -> MetricHistory:
    return MetricHistory(
        metric_key=key,
        title=title,
        data_points=[
            MetricDataPoint(step=s, value=v, timestamp=t) for s, v, t in steps
        ],
    )


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render(table: Table, *, width: int = 120) -> str:
    """Render a Rich Table to plain text for assertion checks."""
    # Rich 14.x derives max_width from terminal env when force_terminal=True.
    # Pin COLUMNS so render width matches the table width under test.
    console = RichConsole(
        width=width,
        no_color=True,
        force_terminal=True,
        _environ={"COLUMNS": str(width), "LINES": "25"},
    )
    with console.capture() as capture:
        console.print(table, end="")
    return _ANSI_RE.sub("", capture.get())


class TestShouldRenderMetricTrends:
    def test_false_when_not_tty(self) -> None:
        m = [_history("A", steps=[(0, 1.0, 0)])]
        assert should_render_metric_trends(False, 120, m) is False

    def test_false_when_terminal_narrower_than_min(self) -> None:
        m = [_history("A", steps=[(0, 1.0, 0)])]
        assert (
            should_render_metric_trends(True, MIN_TREND_TERMINAL_WIDTH - 1, m) is False
        )

    def test_false_when_metrics_empty(self) -> None:
        assert should_render_metric_trends(True, MIN_TREND_TERMINAL_WIDTH, []) is False

    def test_true_when_tty_width_ok_and_metrics_nonempty(self) -> None:
        m = [_history("A", steps=[(0, 1.0, 0)])]
        assert should_render_metric_trends(True, MIN_TREND_TERMINAL_WIDTH, m) is True


class TestRenderMetricTrends:
    def test_empty_metrics_returns_none(self) -> None:
        assert render_metric_trends([]) is None

    def test_returns_rich_table(self) -> None:
        m = [_history("Reward", key="r", steps=[(0, 1.0, 0), (1, 2.0, 0)])]
        result = render_metric_trends(m, terminal_width=120)
        assert isinstance(result, Table)

    def test_empty_series_shows_no_data(self) -> None:
        m = [_history("Train Loss", key="train/loss", steps=[])]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "Train Loss" in out
        assert "No data" in out

    def test_summary_first_to_latest_not_min_max(self) -> None:
        """Non-monotonic series: first/last differ from min/max."""
        pts = [(0, 10.0, 0), (1, 5.0, 0), (2, 20.0, 0)]
        m = [_history("Jagged", key="j", steps=pts)]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "10 -> 20" in out
        assert "5 -> 20" not in out

    def test_series_ordered_by_step_not_input_order(self) -> None:
        """API may return points out of step order; summary/spark follow step."""
        pts = [(2, 100.0, 0), (0, 1.0, 0), (1, 2.0, 0)]
        m = [_history("Step order", key="o", steps=pts)]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "1 -> 100" in out
        assert "100 -> 100" not in out

    def test_blank_rows_between_metrics(self) -> None:
        a = _history("Alpha", key="a", steps=[(0, 1.0, 0)])
        b = _history("Beta", key="b", steps=[(0, 2.0, 0)])
        c = _history("Gamma", key="c", steps=[(0, 3.0, 0)])
        result = render_metric_trends([a, b, c], terminal_width=120)
        out = _render(result)
        lines = out.splitlines()
        # 3 metrics + 2 blank separator rows = 5 lines
        assert len(lines) == 5
        assert "Alpha" in lines[0]
        assert lines[1].strip() == ""
        assert "Beta" in lines[2]
        assert lines[3].strip() == ""
        assert "Gamma" in lines[4]

    def test_flat_series_mid_sparkline_and_summary(self) -> None:
        pts = [(i, 0.5, 0) for i in range(8)]
        m = [_history("Reward", key="r", steps=pts)]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "Reward" in out
        assert "0.500 -> 0.500" in out
        mid = SPARKLINE_BLOCKS[len(SPARKLINE_BLOCKS) // 2]
        assert mid * 8 in out

    def test_metric_order_preserved(self) -> None:
        a = _history("First", key="a", steps=[(0, 1.0, 0)])
        b = _history("Second", key="b", steps=[(0, 2.0, 0)])
        result = render_metric_trends([a, b], terminal_width=120)
        assert result is not None
        out = _render(result)
        assert out.index("First") < out.index("Second")

    def test_integer_like_values_formatted_without_fraction(self) -> None:
        m = [
            _history("Steps", key="s", steps=[(0, 42.0, 0), (1, 100.0, 0)]),
        ]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "42 -> 100" in out

    def test_tiny_non_zero_values_use_compact_form_not_zero_point_three_decimals(
        self,
    ) -> None:
        """Sub-milli non-zero values must not format as 0.000 in the summary."""
        pts = [(0, 1e-6, 0), (1, 2e-6, 0)]
        m = [_history("Tiny", key="t", steps=pts)]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        assert "0.000 -> 0.000" not in out
        assert "1e-06" in out
        assert "2e-06" in out

    def test_long_series_downsampling_is_deterministic(self) -> None:
        pts = [(i, float(i), 0) for i in range(200)]
        m = [_history("Long", key="long", steps=pts)]
        out1 = _render(render_metric_trends(m, terminal_width=120))
        out2 = _render(render_metric_trends(m, terminal_width=120))
        assert out1 == out2
        assert "Long" in out1

    def test_long_series_sparkline_contains_block_chars(self) -> None:
        """Downsampled series produces block characters within terminal width."""
        pts = [(i, float(i), 0) for i in range(200)]
        m = [_history("Wide", key="w", steps=pts)]
        result = render_metric_trends(m, terminal_width=120)
        assert result is not None
        out = _render(result)
        m_spark = re.search(r"[▁▂▃▄▅▆▇█]{8,}", out)
        assert m_spark is not None
        spark = m_spark.group(0)
        allowed = set(SPARKLINE_BLOCKS)
        assert set(spark) <= allowed
        assert len(spark) < len(pts)
        assert len(spark) >= 8

    def test_sparkline_before_summary_on_the_right(self) -> None:
        """Spark fills the middle; summary column is last (right-aligned block)."""
        m = [_history("Col", key="k", steps=[(0, 1.0, 0), (1, 2.0, 0)])]
        result = render_metric_trends(m, terminal_width=100)
        out = _render(result, width=100)
        line = out.splitlines()[0]
        assert "1 -> 2" in line
        first_blk = min(line.find(c) for c in SPARKLINE_BLOCKS if line.find(c) >= 0)
        assert first_blk < line.rindex("1 -> 2")

    def test_opposite_huge_finite_values_do_not_raise(self) -> None:
        """Mixed +/-huge magnitudes must not break sparkline normalization."""
        pts = [(0, -1e308, 0), (1, 1e308, 0)]
        m = [_history("Huge", key="h", steps=pts)]
        result = render_metric_trends(m, terminal_width=MIN_TREND_TERMINAL_WIDTH)
        assert isinstance(result, Table)
        out = _render(result, width=MIN_TREND_TERMINAL_WIDTH)
        assert "Huge" in out

    def test_non_finite_and_extreme_values_render_without_error(self) -> None:
        """Pathological floats must not raise; output stays reasonable."""
        pts = [
            (0, float("nan"), 0),
            (1, float("inf"), 1),
            (2, float("-inf"), 2),
            (3, 1e300, 3),
            (4, 0.5, 4),
        ]
        m = [_history("Edge", key="e", steps=pts)]
        result = render_metric_trends(m, terminal_width=MIN_TREND_TERMINAL_WIDTH)
        assert isinstance(result, Table)
        out = _render(result, width=MIN_TREND_TERMINAL_WIDTH)
        assert out
        assert "Edge" in out


class TestDownsampleValues:
    def test_single_sample_uses_latest_point(self) -> None:
        assert _downsample_values([1.0, 2.0, 99.0], 1) == [99.0]
