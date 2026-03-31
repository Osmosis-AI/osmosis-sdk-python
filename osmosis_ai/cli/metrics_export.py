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
}


def _epoch_ms_to_iso(epoch_ms: int) -> str:
    """Convert epoch milliseconds to ISO 8601 string."""
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000, tz=datetime.timezone.utc)
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
        training_run["model_name"] = run.model_name
    if metrics.overview.duration_formatted is not None:
        training_run["duration"] = metrics.overview.duration_formatted
    if run.started_at is not None:
        training_run["started_at"] = run.started_at
    if run.completed_at is not None:
        training_run["completed_at"] = run.completed_at
    if run.examples_processed_count is not None:
        training_run["examples_processed"] = run.examples_processed_count

    # summary section
    summary: dict[str, Any] = {
        "total_steps": max(
            (dp.step for m in metrics.metrics for dp in m.data_points), default=0
        ),
    }
    if metrics.overview.reward is not None:
        summary["final_reward"] = metrics.overview.reward
    if metrics.overview.reward_delta is not None:
        summary["reward_delta"] = metrics.overview.reward_delta

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
