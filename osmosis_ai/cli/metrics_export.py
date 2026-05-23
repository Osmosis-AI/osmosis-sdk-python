"""Transform training run metrics API response into export JSON format."""

from __future__ import annotations

import datetime
from typing import Any

from osmosis_ai.platform.api.models import TrainingRunDetail, TrainingRunMetrics

# MLflow internal key → human-readable export key
_METRIC_KEY_MAP: dict[str, str] = {
    "rollout/raw_reward": "training_reward",
    "eval/validation/reward": "validation_reward",
    "train/entropy_loss": "entropy",
    "rollout/response_lengths": "response_length",
    "rollout/total_lengths": "total_length",
    "rollout/truncated_ratio": "truncation_ratio",
}


def _epoch_ms_to_iso(epoch_ms: int) -> str:
    """Convert epoch milliseconds to ISO 8601 string."""
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000, tz=datetime.UTC)
    return dt.isoformat()


def build_export_dict(
    run: TrainingRunDetail,
    metrics: TrainingRunMetrics,
) -> dict[str, Any]:
    """Build the export JSON dict from API response models.

    Performs metric key mapping, timestamp conversion, and null omission.
    """
    # training_run section — omit None values
    training_run: dict[str, Any] = {
        "id": run.id,
        "status": run.status,
    }
    if run.name is not None:
        training_run["name"] = run.name
    if run.model_name is not None:
        training_run["base_model_name"] = run.model_name
    if run.dataset_name is not None:
        training_run["dataset_name"] = run.dataset_name
    if run.rollout_name is not None:
        training_run["rollout_name"] = run.rollout_name
    if metrics.overview.duration_formatted is not None:
        training_run["duration"] = metrics.overview.duration_formatted
    if run.started_at is not None:
        training_run["started_at"] = run.started_at
    if run.completed_at is not None:
        training_run["completed_at"] = run.completed_at
    if run.examples_processed_count is not None:
        training_run["rows_processed"] = run.examples_processed_count
    if metrics.overview.latest_step is not None:
        training_run["latest_step"] = metrics.overview.latest_step
    if metrics.overview.total_steps is not None:
        training_run["total_steps"] = metrics.overview.total_steps

    # summary section
    summary: dict[str, Any] = {}
    for s in metrics.overview.metric_summaries:
        key = _METRIC_KEY_MAP.get(s.key, s.key)
        entry: dict[str, float] = {}
        if s.initial is not None:
            entry["initial"] = s.initial
        if s.latest is not None:
            entry["latest"] = s.latest
        if s.delta is not None:
            entry["delta"] = s.delta
        if s.min is not None:
            entry["min"] = s.min
        if s.max is not None:
            entry["max"] = s.max
        if entry:
            summary[key] = entry

    # metrics section
    export_metrics = []
    for m in metrics.metrics:
        key = _METRIC_KEY_MAP.get(m.metric_key, m.metric_key)
        data = [
            {
                "step": dp.step,
                "value": dp.value,
                "timestamp": _epoch_ms_to_iso(dp.timestamp),
            }
            for dp in m.data_points
        ]
        export_metrics.append(
            {
                "key": key,
                "title": m.title,
                "data": data,
            }
        )

    return {
        "training_run": training_run,
        "summary": summary,
        "metrics": export_metrics,
    }
