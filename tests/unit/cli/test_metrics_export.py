"""Tests for osmosis_ai.cli.metrics_export."""

from __future__ import annotations

from osmosis_ai.cli.metrics_export import build_eval_export_dict, build_export_dict
from osmosis_ai.platform.api.models import (
    EvalPassAtKPoint,
    EvalRewardStats,
    EvalRunMetrics,
    EvalRunMetricsOverview,
    EvaluationRunDetail,
    MetricDataPoint,
    MetricHistory,
    MetricSummary,
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
                duration_ms=3600000,
                metric_summaries=[
                    MetricSummary(
                        key="rollout/raw_reward",
                        title="Training Reward",
                        initial=0.70,
                        latest=0.85,
                        delta=0.15,
                        min=0.65,
                        max=0.87,
                    ),
                ],
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
            dataset_name="my-dataset.jsonl",
            rollout_name="my-rollout",
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
        assert tr["base_model_name"] == "Qwen/Qwen3-8B"
        assert tr["started_at"] == "2026-03-28T10:00:00Z"
        assert tr["completed_at"] == "2026-03-28T11:05:30Z"
        assert tr["rows_processed"] == 5000
        assert tr["dataset_name"] == "my-dataset.jsonl"
        assert tr["rollout_name"] == "my-rollout"

    def test_summary_section(self) -> None:
        metrics = self._make_metrics()
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        summary = result["summary"]
        assert summary["training_reward"]["latest"] == 0.85
        assert summary["training_reward"]["delta"] == 0.15
        assert summary["training_reward"]["initial"] == 0.70

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
        assert "base_model_name" not in tr
        assert "started_at" not in tr
        assert "completed_at" not in tr
        assert "dataset_name" not in tr
        assert "rollout_name" not in tr

    def test_duration_from_overview(self) -> None:
        metrics = self._make_metrics()
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        assert result["training_run"]["duration_ms"] == 3600000

    def test_empty_metrics_list(self) -> None:
        run = self._make_run_detail()
        metrics = TrainingRunMetrics(
            training_run_id="run-1",
            status="finished",
            overview=TrainingRunMetricsOverview(
                duration_ms=None,
                metric_summaries=[],
                examples_processed_count=None,
            ),
            metrics=[],
        )
        result = build_export_dict(run, metrics)
        assert result["metrics"] == []
        assert result["summary"] == {}

    def test_summary_includes_all_metric_summaries(self) -> None:
        run = self._make_run_detail()
        metrics = TrainingRunMetrics(
            training_run_id="run-1",
            status="finished",
            overview=TrainingRunMetricsOverview(
                duration_ms=None,
                metric_summaries=[
                    MetricSummary(
                        key="rollout/raw_reward",
                        title="Training Reward",
                        initial=0.70,
                        latest=0.85,
                        delta=0.15,
                        min=0.65,
                        max=0.87,
                    ),
                ],
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
        assert result["summary"]["training_reward"]["latest"] == 0.85
        assert result["summary"]["training_reward"]["delta"] == 0.15

    def test_step_progress_from_overview(self) -> None:
        metrics = self._make_metrics()
        metrics.overview.latest_step = 120
        metrics.overview.total_steps = 200
        run = self._make_run_detail()
        result = build_export_dict(run, metrics)
        tr = result["training_run"]
        assert tr["latest_step"] == 120
        assert tr["total_steps"] == 200


class TestBuildEvalExportDict:
    """Tests for build_eval_export_dict — the eval API-to-JSON transformer."""

    def _make_detail(self) -> EvaluationRunDetail:
        return EvaluationRunDetail(
            id="eval_1",
            name="math-eval",
            status="succeeded",
            created_at="2026-05-04T00:00:00Z",
            started_at="2026-05-04T00:01:00Z",
            completed_at="2026-05-04T00:31:00Z",
            model={"name": "openai/gpt-5-mini"},
            dataset={"file_name": "eval.jsonl"},
            rollout={"model_name": "math-rollout"},
        )

    def _make_metrics(
        self,
        *,
        reward_stats: EvalRewardStats | None = None,
        pass_at_k: list[EvalPassAtKPoint] | None = None,
    ) -> EvalRunMetrics:
        return EvalRunMetrics(
            eval_run_id="eval_1",
            status="succeeded",
            overview=EvalRunMetricsOverview(
                duration_ms=1800000,
                total_samples=100,
                completed_samples=100,
                graded=98,
                passed=80,
                failed=18,
                skipped=2,
                pass_rate=0.8163,
                pass_threshold=0.5,
                tokens_used=250000,
            ),
            reward_stats=reward_stats,
            pass_at_k=pass_at_k or [],
        )

    def test_eval_run_section(self) -> None:
        result = build_eval_export_dict(self._make_detail(), self._make_metrics())
        er = result["eval_run"]
        assert er["id"] == "eval_1"
        assert er["status"] == "succeeded"
        assert er["name"] == "math-eval"
        assert er["model_name"] == "openai/gpt-5-mini"
        assert er["dataset_name"] == "eval.jsonl"
        assert er["rollout_name"] == "math-rollout"
        assert er["duration_ms"] == 1800000
        assert er["started_at"] == "2026-05-04T00:01:00Z"
        assert er["completed_at"] == "2026-05-04T00:31:00Z"

    def test_summary_section(self) -> None:
        result = build_eval_export_dict(self._make_detail(), self._make_metrics())
        summary = result["summary"]
        assert summary["total_samples"] == 100
        assert summary["completed_samples"] == 100
        assert summary["graded"] == 98
        assert summary["passed"] == 80
        assert summary["failed"] == 18
        assert summary["skipped"] == 2
        assert summary["pass_rate"] == 0.8163
        assert summary["pass_threshold"] == 0.5
        assert summary["tokens_used"] == 250000
        assert "reward_stats" not in summary
        assert "pass_at_k" not in summary

    def test_null_fields_omitted(self) -> None:
        detail = EvaluationRunDetail(
            id="eval_1",
            name=None,
            status="pending",
            created_at="2026-05-04T00:00:00Z",
        )
        metrics = EvalRunMetrics(
            eval_run_id="eval_1",
            status="pending",
            overview=EvalRunMetricsOverview(
                duration_ms=None,
                total_samples=None,
                completed_samples=None,
                graded=None,
                passed=None,
                failed=None,
                skipped=None,
                pass_rate=None,
                pass_threshold=None,
                tokens_used=None,
            ),
            reward_stats=None,
            pass_at_k=[],
        )
        result = build_eval_export_dict(detail, metrics)
        assert result["eval_run"] == {"id": "eval_1", "status": "pending"}
        assert result["summary"] == {}

    def test_ref_name_ignores_non_string_values(self) -> None:
        detail = self._make_detail()
        detail.model = {"name": 123}
        detail.dataset = {}
        detail.rollout = {"name": ""}
        result = build_eval_export_dict(detail, self._make_metrics())
        er = result["eval_run"]
        assert "model_name" not in er
        assert "dataset_name" not in er
        assert "rollout_name" not in er

    def test_reward_stats_included(self) -> None:
        metrics = self._make_metrics(
            reward_stats=EvalRewardStats(
                mean=0.72, median=0.75, std=0.11, min=0.2, max=0.98
            )
        )
        result = build_eval_export_dict(self._make_detail(), metrics)
        assert result["summary"]["reward_stats"] == {
            "mean": 0.72,
            "median": 0.75,
            "std": 0.11,
            "min": 0.2,
            "max": 0.98,
        }

    def test_reward_stats_omits_none_values(self) -> None:
        metrics = self._make_metrics(
            reward_stats=EvalRewardStats(
                mean=0.72, median=None, std=None, min=None, max=None
            )
        )
        result = build_eval_export_dict(self._make_detail(), metrics)
        assert result["summary"]["reward_stats"] == {"mean": 0.72}

    def test_reward_stats_omitted_when_all_none(self) -> None:
        metrics = self._make_metrics(
            reward_stats=EvalRewardStats(
                mean=None, median=None, std=None, min=None, max=None
            )
        )
        result = build_eval_export_dict(self._make_detail(), metrics)
        assert "reward_stats" not in result["summary"]

    def test_pass_at_k_included(self) -> None:
        metrics = self._make_metrics(
            pass_at_k=[
                EvalPassAtKPoint(k=1, value=0.6),
                EvalPassAtKPoint(k=4, value=0.85),
            ]
        )
        result = build_eval_export_dict(self._make_detail(), metrics)
        assert result["summary"]["pass_at_k"] == [
            {"k": 1, "value": 0.6},
            {"k": 4, "value": 0.85},
        ]
