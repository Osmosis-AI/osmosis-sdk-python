"""Tests for osmosis_ai.cli.metrics_export."""

from __future__ import annotations

from osmosis_ai.cli.metrics_export import build_export_dict
from osmosis_ai.platform.api.models import (
    MetricDataPoint,
    MetricHistory,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)


class TestBuildExportDict:
    """Tests for build_export_dict — the API-to-JSON transformer."""

    def _make_metrics(
        self,
        *,
        metric_key: str = "rollout/raw_reward",
        title: str = "Training Reward",
        steps: list[tuple[int, float, int]] | None = None,
    ) -> TrainingRunMetrics:
        if steps is None:
            steps = [(0, 0.5, 1711800000000), (10, 0.65, 1711800060000)]
        return TrainingRunMetrics(
            training_run_id="550e8400-e29b-41d4-a716-446655440000",
            status="finished",
            overview=TrainingRunMetricsOverview(
                mlflow_run_id="mlflow-abc",
                mlflow_status="FINISHED",
                duration_ms=3600000,
                duration_formatted="1h",
                reward=0.85,
                reward_delta=0.15,
                examples_processed_count=5000,
            ),
            metrics=[
                MetricHistory(
                    metric_key=metric_key,
                    title=title,
                    data_points=[
                        MetricDataPoint(step=s, value=v, timestamp=t)
                        for s, v, t in steps
                    ],
                ),
            ],
        )

    def _make_run_detail(self) -> TrainingRunDetail:
        return TrainingRunDetail(
            id="550e8400-e29b-41d4-a716-446655440000",
            name="reward-tuning-v3",
            status="finished",
            model_name="Qwen/Qwen3-8B",
            started_at="2026-03-28T10:00:00Z",
            completed_at="2026-03-28T11:05:30Z",
            examples_processed_count=5000,
        )

    def test_training_run_section(self) -> None:
        metrics = self._make_metrics()
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        tr = result["training_run"]
        assert tr["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert tr["name"] == "reward-tuning-v3"
        assert tr["status"] == "finished"
        assert tr["model_name"] == "Qwen/Qwen3-8B"
        assert tr["started_at"] == "2026-03-28T10:00:00Z"
        assert tr["completed_at"] == "2026-03-28T11:05:30Z"
        assert tr["examples_processed"] == 5000

    def test_summary_section(self) -> None:
        metrics = self._make_metrics()
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        summary = result["summary"]
        assert summary["final_reward"] == 0.85
        assert summary["reward_delta"] == 0.15
        assert summary["total_steps"] == 10

    def test_metric_key_mapping(self) -> None:
        metrics = self._make_metrics(
            metric_key="rollout/raw_reward", title="Training Reward"
        )
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        m = result["metrics"][0]
        assert m["key"] == "training_reward"
        assert m["title"] == "Training Reward"

    def test_timestamp_converted_to_iso(self) -> None:
        metrics = self._make_metrics(steps=[(0, 0.5, 1711800000000)])
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        point = result["metrics"][0]["data"][0]
        assert point["step"] == 0
        assert point["value"] == 0.5
        # epoch 1711800000000 ms = 2024-03-30T12:00:00Z
        assert point["timestamp"] == "2024-03-30T12:00:00+00:00"

    def test_unknown_metric_key_uses_original(self) -> None:
        metrics = self._make_metrics(
            metric_key="custom/my_metric", title="My Custom Metric"
        )
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        m = result["metrics"][0]
        assert m["key"] == "custom/my_metric"

    def test_null_fields_omitted(self) -> None:
        run = TrainingRunDetail(
            id="run-1",
            name=None,
            status="finished",
            model_name=None,
            started_at=None,
            completed_at=None,
        )
        metrics = self._make_metrics(steps=[])
        result = build_export_dict(run, metrics)
        tr = result["training_run"]
        assert "name" not in tr
        assert "model_name" not in tr
        assert "started_at" not in tr
        assert "completed_at" not in tr

    def test_duration_from_overview(self) -> None:
        metrics = self._make_metrics()
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        assert result["training_run"]["duration"] == "1h"

    def test_empty_metrics_list(self) -> None:
        run = self._make_run_detail()
        metrics = TrainingRunMetrics(
            training_run_id="run-1",
            status="finished",
            overview=TrainingRunMetricsOverview(
                mlflow_run_id="m",
                mlflow_status="FINISHED",
                duration_ms=None,
                duration_formatted=None,
                reward=None,
                reward_delta=None,
                examples_processed_count=None,
            ),
            metrics=[],
        )
        result = build_export_dict(run, metrics)
        assert result["metrics"] == []
        assert result["summary"]["total_steps"] == 0

    def test_total_steps_max_across_metrics(self) -> None:
        run = self._make_run_detail()
        metrics = TrainingRunMetrics(
            training_run_id="run-1",
            status="finished",
            overview=TrainingRunMetricsOverview(
                mlflow_run_id="m",
                mlflow_status="FINISHED",
                duration_ms=None,
                duration_formatted=None,
                reward=0.85,
                reward_delta=0.15,
                examples_processed_count=None,
            ),
            metrics=[
                MetricHistory(
                    metric_key="rollout/raw_reward",
                    title="Training Reward",
                    data_points=[
                        MetricDataPoint(step=0, value=0.5, timestamp=1711800000000),
                        MetricDataPoint(step=100, value=0.8, timestamp=1711800060000),
                    ],
                ),
                MetricHistory(
                    metric_key="eval/validation/reward",
                    title="Validation Reward",
                    data_points=[
                        MetricDataPoint(step=0, value=0.4, timestamp=1711800000000),
                        MetricDataPoint(step=500, value=0.7, timestamp=1711800060000),
                    ],
                ),
            ],
        )
        result = build_export_dict(run, metrics)
        assert result["summary"]["total_steps"] == 500
