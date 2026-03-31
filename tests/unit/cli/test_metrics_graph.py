"""Tests for osmosis_ai.cli.metrics_graph — terminal metric trend rendering."""

from __future__ import annotations

import re

from osmosis_ai.cli.metrics_graph import (
    MIN_TREND_TERMINAL_WIDTH,
    SPARKLINE_BLOCKS,
    _downsample_values,
    _layout_columns,
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

    def test_empty_series_shows_no_data(self) -> None:
        m = [_history("Train Loss", key="train/loss", steps=[])]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert out.splitlines() == ["Train Loss  No data"]

    def test_summary_first_to_latest_not_min_max(self) -> None:
        """Non-monotonic series: first/last differ from min/max."""
        pts = [(0, 10.0, 0), (1, 5.0, 0), (2, 20.0, 0)]
        m = [_history("Jagged", key="j", steps=pts)]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert "10 -> 20" in out
        assert "5 -> 20" not in out

    def test_series_ordered_by_step_not_input_order(self) -> None:
        """API may return points out of step order; summary/spark follow step."""
        pts = [(2, 100.0, 0), (0, 1.0, 0), (1, 2.0, 0)]
        m = [_history("Step order", key="o", steps=pts)]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert "1 -> 100" in out
        assert "100 -> 100" not in out

    def test_one_line_per_metric(self) -> None:
        a = _history("Alpha", key="a", steps=[(0, 1.0, 0)])
        b = _history("Beta", key="b", steps=[(0, 2.0, 0)])
        out = render_metric_trends([a, b], terminal_width=120)
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("Alpha  ")
        assert lines[1].startswith("Beta  ")

    def test_flat_series_mid_sparkline_and_summary(self) -> None:
        pts = [(i, 0.5, 0) for i in range(8)]
        m = [_history("Reward", key="r", steps=pts)]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert "Reward" in out
        assert "0.500 -> 0.500" in out
        mid = SPARKLINE_BLOCKS[len(SPARKLINE_BLOCKS) // 2]
        assert mid * 8 in out

    def test_metric_order_preserved(self) -> None:
        a = _history("First", key="a", steps=[(0, 1.0, 0)])
        b = _history("Second", key="b", steps=[(0, 2.0, 0)])
        out = render_metric_trends([a, b], terminal_width=120)
        assert out is not None
        assert out.index("First") < out.index("Second")

    def test_integer_like_values_formatted_without_fraction(self) -> None:
        m = [
            _history("Steps", key="s", steps=[(0, 42.0, 0), (1, 100.0, 0)]),
        ]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert "42 -> 100" in out

    def test_tiny_non_zero_values_use_compact_form_not_zero_point_three_decimals(
        self,
    ) -> None:
        """Sub-milli non-zero values must not format as 0.000 in the summary."""
        pts = [(0, 1e-6, 0), (1, 2e-6, 0)]
        m = [_history("Tiny", key="t", steps=pts)]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        assert "0.000 -> 0.000" not in out
        assert "1e-06" in out
        assert "2e-06" in out

    def test_long_series_downsampling_is_deterministic(self) -> None:
        pts = [(i, float(i), 0) for i in range(200)]
        m = [_history("Long", key="long", steps=pts)]
        out1 = render_metric_trends(m, terminal_width=120)
        out2 = render_metric_trends(m, terminal_width=120)
        assert out1 == out2
        assert out1 is not None
        assert "Long" in out1

    def test_long_series_sparkline_shorter_than_raw_points(self) -> None:
        """Downsampling keeps output bounded without pinning internal width constants."""
        pts = [(i, float(i), 0) for i in range(200)]
        m = [_history("Wide", key="w", steps=pts)]
        out = render_metric_trends(m, terminal_width=120)
        assert out is not None
        line = out.splitlines()[0]
        m_spark = re.search(r"[▁▂▃▄▅▆▇█]{8,}", line)
        assert m_spark is not None
        spark = m_spark.group(0)
        allowed = set(SPARKLINE_BLOCKS)
        assert set(spark) <= allowed
        assert len(spark) < len(pts)
        assert len(spark) >= 8

    def test_row_layout_sparkline_before_summary_on_the_right(self) -> None:
        """Spark fills the middle; summary column is last (right-aligned block)."""
        m = [_history("Col", key="k", steps=[(0, 1.0, 0), (1, 2.0, 0)])]
        out = render_metric_trends(m, terminal_width=100)
        line = out.splitlines()[0]
        assert line.endswith("1 -> 2")
        first_blk = min(line.find(c) for c in SPARKLINE_BLOCKS if line.find(c) >= 0)
        assert first_blk < line.rindex("1 -> 2")
        assert re.search(r"[▁▂▃▄▅▆▇█]{2,}  +.*->", line)

    def test_sparkline_padded_so_summary_column_aligns_across_rows(self) -> None:
        """Short and long series share the same summary column start (stable layout)."""
        short_m = _history("Short", key="s", steps=[(0, 1.0, 0), (1, 2.0, 0)])
        long_pts = [(i, float(i), 0) for i in range(80)]
        long_m = _history("Longer", key="l", steps=long_pts)
        tw = 120
        metrics = [short_m, long_m]
        title_col, summary_col, spark_w = _layout_columns(metrics, tw)
        summary_start = title_col + 2 + spark_w + 2
        spark_start = title_col + 2
        out = render_metric_trends(metrics, terminal_width=tw)
        lines = out.splitlines()
        assert len(lines) == 2
        assert all(len(line) == tw for line in lines)
        for line in lines:
            assert len(line[spark_start : spark_start + spark_w]) == spark_w
            assert len(line[summary_start : summary_start + summary_col]) == summary_col

    def test_opposite_huge_finite_values_do_not_raise(self) -> None:
        """Mixed ±huge magnitudes must not break sparkline normalization."""
        pts = [(0, -1e308, 0), (1, 1e308, 0)]
        m = [_history("Huge", key="h", steps=pts)]
        out = render_metric_trends(m, terminal_width=MIN_TREND_TERMINAL_WIDTH)
        assert isinstance(out, str)
        assert "Huge" in out

    def test_lines_never_exceed_terminal_width_with_long_title(self) -> None:
        """Title + spark + summary must fit; long titles shrink spark/title, not wrap."""
        tw = MIN_TREND_TERMINAL_WIDTH
        long_title = "M" * 52
        m = [
            _history(long_title, key="k", steps=[(0, 1.0, 0), (1, 2.0, 0)]),
        ]
        out = render_metric_trends(m, terminal_width=tw)
        assert out is not None
        assert all(len(line) <= tw for line in out.splitlines())

    def test_non_finite_and_extreme_values_render_without_error(self) -> None:
        """Pathological floats must not raise; output stays within terminal width."""
        pts = [
            (0, float("nan"), 0),
            (1, float("inf"), 1),
            (2, float("-inf"), 2),
            (3, 1e300, 3),
            (4, 0.5, 4),
        ]
        m = [_history("Edge", key="e", steps=pts)]
        tw = MIN_TREND_TERMINAL_WIDTH
        out = render_metric_trends(m, terminal_width=tw)
        assert isinstance(out, str)
        assert out
        assert "Edge" in out
        for line in out.splitlines():
            assert len(line) <= tw


class TestDownsampleValues:
    def test_single_sample_uses_latest_point(self) -> None:
        assert _downsample_values([1.0, 2.0, 99.0], 1) == [99.0]
