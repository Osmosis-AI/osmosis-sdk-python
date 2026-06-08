"""Transform run metrics API responses into export JSON format."""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.paths import parse_cli_path
from osmosis_ai.platform.api.models import (
    EvalRunMetrics,
    EvaluationRunDetail,
    TrainingRunDetail,
    TrainingRunMetrics,
)


def safe_name(name: str) -> str:
    """Sanitise a run name for use as a filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def default_metrics_filename(run_name: str | None, run_id: str) -> str:
    """Build the default JSON filename from run metadata."""
    short_id = run_id[:8]
    safe = safe_name(run_name) if run_name else None
    return f"{safe}_{short_id}.json" if safe else f"{short_id}.json"


def resolve_metrics_output_path(output: str, run_name: str | None, run_id: str) -> Path:
    """Resolve a user-supplied ``-o`` value into a concrete file path.

    Rules:
    * Trailing ``/`` or existing directory → directory mode (generate default
      filename inside the directory).
    * Has a file extension → use as-is.
    * No extension → auto-append ``.json``.

    Parent directories are created automatically.
    """
    parsed_output = parse_cli_path(output)
    path = parsed_output.path

    try:
        if parsed_output.has_trailing_separator or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            return path / default_metrics_filename(run_name, run_id)

        if path.suffix != ".json":
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(f"Cannot create output path: {exc}") from exc

    return path


def resolve_default_metrics_output(
    run_name: str | None, run_id: str, *, workspace_directory: Path
) -> Path:
    """Resolve the default output path under .osmosis/metrics/."""
    metrics_dir = workspace_directory / ".osmosis" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / default_metrics_filename(run_name, run_id)


def _ref_name(ref: dict[str, Any] | None) -> str | None:
    if not ref:
        return None
    value = ref.get("name") or ref.get("file_name") or ref.get("model_name")
    return value if isinstance(value, str) else None


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
    if metrics.overview.duration_ms is not None:
        training_run["duration_ms"] = metrics.overview.duration_ms
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


def build_eval_export_dict(
    detail: EvaluationRunDetail,
    metrics: EvalRunMetrics,
) -> dict[str, Any]:
    """Build the eval export JSON dict from API response models.

    Combines run identity from the detail with the metrics summary, omitting
    None values.
    """
    overview = metrics.overview

    eval_run: dict[str, Any] = {"id": detail.id, "status": detail.status}
    if detail.name is not None:
        eval_run["name"] = detail.name
    dataset_name = _ref_name(detail.dataset)
    if dataset_name is not None:
        eval_run["dataset_name"] = dataset_name
    model_name = _ref_name(detail.model)
    if model_name is not None:
        eval_run["model_name"] = model_name
    rollout_name = _ref_name(detail.rollout)
    if rollout_name is not None:
        eval_run["rollout_name"] = rollout_name
    if overview.duration_ms is not None:
        eval_run["duration_ms"] = overview.duration_ms
    if detail.started_at is not None:
        eval_run["started_at"] = detail.started_at
    if detail.completed_at is not None:
        eval_run["completed_at"] = detail.completed_at

    summary: dict[str, Any] = {}
    summary_fields: list[tuple[str, Any]] = [
        ("total_samples", overview.total_samples),
        ("completed_samples", overview.completed_samples),
        ("graded", overview.graded),
        ("passed", overview.passed),
        ("failed", overview.failed),
        ("skipped", overview.skipped),
        ("pass_rate", overview.pass_rate),
        ("pass_threshold", overview.pass_threshold),
        ("tokens_used", overview.tokens_used),
    ]
    for key, value in summary_fields:
        if value is not None:
            summary[key] = value

    if metrics.reward_stats is not None:
        rs = metrics.reward_stats
        reward_stats = {
            stat: value
            for stat, value in (
                ("mean", rs.mean),
                ("median", rs.median),
                ("std", rs.std),
                ("min", rs.min),
                ("max", rs.max),
            )
            if value is not None
        }
        if reward_stats:
            summary["reward_stats"] = reward_stats

    if metrics.pass_at_k:
        summary["pass_at_k"] = [{"k": p.k, "value": p.value} for p in metrics.pass_at_k]

    return {"eval_run": eval_run, "summary": summary}
