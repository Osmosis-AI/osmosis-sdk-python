"""Best-effort saving of finished rollouts as ATIF trajectory documents.

Documents land at ``<artifact_root>/<rollout_id>/trajectory.json``, next to
the rollout's file artifacts; backend diagnostics land alongside as
``diagnostics.json``. Failures are logged and swallowed.
"""

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harbor.models.trajectories import FinalMetrics, Trajectory
from harbor.utils.trajectory_utils import format_trajectory_json

from osmosis_ai.rollout.trajectory.converter import (
    _apply_report,
    _final_metrics_from_report,
    convert_sample_to_trajectory,
)
from osmosis_ai.rollout.trajectory.report import SampleReport, TrajectoryReport
from osmosis_ai.rollout.types import ExecutionResult
from osmosis_ai.rollout.utils.file_artifacts import default_artifact_root

logger: logging.Logger = logging.getLogger(__name__)

# Sample token totals (harbor's per-trial accounting) seed FinalMetrics
# fields a native document left unset.
_FINAL_METRIC_BY_SAMPLE_KEY = {
    "input_tokens": "total_prompt_tokens",
    "output_tokens": "total_completion_tokens",
    "cached_tokens": "total_cached_tokens",
    "cost_usd": "total_cost_usd",
}


def _write_document(dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def _resolve_sample_report(
    report: TrajectoryReport | None,
) -> tuple[SampleReport | None, dict[str, SampleReport]]:
    """Pick the report entry for the rollout's single sample.

    Sample ids no longer exist on the SDK side, so a lone entry applies
    regardless of its key. Multiple entries cannot be attributed and are
    returned for preservation under ``extra``.
    """
    if report is None or not report.samples:
        return None, {}
    if len(report.samples) == 1:
        return next(iter(report.samples.values())), {}
    return None, dict(report.samples)


def _prepare_native_trajectory(
    document: dict[str, Any],
    *,
    rollout_id: str,
    result: ExecutionResult,
    request_label: str | None,
    request_metadata: dict[str, Any] | None,
    request_extra_fields: dict[str, Any] | None,
    report: TrajectoryReport | None,
) -> Trajectory:
    """Enrich a backend-native ATIF document without rebuilding its steps.

    The agent's own step structure — tool calls, observations, reasoning,
    token ids, logprobs, tool definitions, subagent trajectories — stays
    authoritative; controller-reported metrics and Osmosis rollout metadata
    are overlaid on top.
    """
    trajectory = Trajectory.model_validate(document)
    native_session_id = trajectory.session_id
    native_trajectory_id = trajectory.trajectory_id

    matched_report, unmatched_reports = _resolve_sample_report(report)
    unmatched_llm_call_metrics = _apply_report(trajectory.steps, matched_report)

    sample = result.sample
    if trajectory.final_metrics is None:
        trajectory.final_metrics = FinalMetrics()
    # Seed totals harbor already accounted for the trial when the document
    # carries none; the controller's report still wins below.
    sample_metrics = sample.metrics if sample is not None else {}
    for sample_key, final_field in _FINAL_METRIC_BY_SAMPLE_KEY.items():
        value = sample_metrics.get(sample_key)
        if value is not None and getattr(trajectory.final_metrics, final_field) is None:
            setattr(trajectory.final_metrics, final_field, value)
    controller_metrics = _final_metrics_from_report(matched_report)
    if controller_metrics is not None:
        for field, value in controller_metrics.model_dump(exclude_none=True).items():
            if field != "total_steps":
                setattr(trajectory.final_metrics, field, value)
    trajectory.final_metrics.total_steps = len(trajectory.steps)

    reported_model = (
        matched_report.model_name if matched_report is not None else None
    ) or (report.model_name if report is not None else None)
    if reported_model:
        trajectory.agent.model_name = reported_model

    # Normalize ids for platform joins; harbor's own stay under extra.
    trajectory.session_id = rollout_id
    trajectory.trajectory_id = rollout_id

    extra = dict(trajectory.extra or {})
    existing_osmosis = extra.get("osmosis")
    osmosis = dict(existing_osmosis) if isinstance(existing_osmosis, Mapping) else {}
    updates = {
        "rollout_id": rollout_id,
        "native_session_id": native_session_id
        if native_session_id != rollout_id
        else None,
        "native_trajectory_id": native_trajectory_id
        if native_trajectory_id not in (None, rollout_id)
        else None,
        "label": sample.label
        if sample is not None and sample.label is not None
        else request_label,
        "reward": sample.reward if sample is not None else None,
        "sample_metrics": sample.metrics if sample and sample.metrics else None,
        "sample_extra_fields": sample.extra_fields
        if sample and sample.extra_fields
        else None,
        "request_metadata": request_metadata,
        "request_extra_fields": request_extra_fields,
        "result_extra_fields": result.extra_fields,
        "unmatched_llm_call_metrics": unmatched_llm_call_metrics,
        "unmatched_sample_reports": {
            key: value.model_dump(exclude_none=True)
            for key, value in unmatched_reports.items()
        }
        or None,
    }
    osmosis.update({key: value for key, value in updates.items() if value is not None})
    extra["osmosis"] = osmosis
    trajectory.extra = extra
    return trajectory


async def save_trajectories(
    *,
    rollout_id: str,
    result: ExecutionResult,
    request_label: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    request_extra_fields: dict[str, Any] | None = None,
    report: TrajectoryReport | None = None,
    artifact_root: Path | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    """Save the rollout's sample as an ATIF document. Never raises.

    ``diagnostics`` overrides ``result.extra_fields`` for the diagnostics
    sidecar, so the caller can retain the latest diagnostics payload
    independently of which result carries the archived sample.
    """
    try:
        await _save(
            rollout_id=rollout_id,
            result=result,
            request_label=request_label,
            request_metadata=request_metadata,
            request_extra_fields=request_extra_fields,
            report=report,
            artifact_root=artifact_root or default_artifact_root(),
            diagnostics=diagnostics,
        )
    except Exception:
        logger.warning(
            "Failed to save trajectories for rollout %s (best-effort)",
            rollout_id,
            exc_info=True,
        )


async def _save(
    *,
    rollout_id: str,
    result: ExecutionResult,
    request_label: str | None,
    request_metadata: dict[str, Any] | None,
    request_extra_fields: dict[str, Any] | None,
    report: TrajectoryReport | None,
    artifact_root: Path,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    # Diagnostics are written before the sample-None early return: a failed
    # rollout with no sample must still leave a durable record on disk.
    payload = diagnostics if diagnostics is not None else result.extra_fields
    if payload is not None:
        diagnostics_dest = artifact_root / rollout_id / "diagnostics.json"
        diagnostics_data = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
        ).encode()
        await asyncio.to_thread(_write_document, diagnostics_dest, diagnostics_data)
        logger.info(
            "Saved rollout diagnostics for %s -> %s", rollout_id, diagnostics_dest
        )

    if result.trajectory_document is not None:
        # A backend-native ATIF document is the authoritative training
        # trajectory; archive it enriched, never rebuilt from messages.
        trajectory = _prepare_native_trajectory(
            result.trajectory_document,
            rollout_id=rollout_id,
            result=result,
            request_label=request_label,
            request_metadata=request_metadata,
            request_extra_fields=request_extra_fields,
            report=report,
        )
        dest = artifact_root / rollout_id / "trajectory.json"
        data = format_trajectory_json(trajectory.to_json_dict()).encode()
        await asyncio.to_thread(_write_document, dest, data)
        logger.info(
            "Saved backend-native trajectory document for rollout %s -> %s",
            rollout_id,
            dest,
        )
        return

    sample = result.sample
    if sample is None:
        return
    if sample.trajectory_messages is None:
        # Explicit opt-out, or an upstream conversion/snapshot failure
        # that already warned with a traceback -- not worth a warning here.
        logger.info(
            "Skipping trajectory for rollout %s: no trajectory messages "
            "(persistence disabled or conversion failed upstream)",
            rollout_id,
        )
        return

    matched_report, unmatched_reports = _resolve_sample_report(report)
    if unmatched_reports:
        logger.warning(
            "Trajectory report for rollout %s has %d entries but the rollout "
            "produced one sample; preserving them under "
            "extra.osmosis.unmatched_sample_reports",
            rollout_id,
            len(unmatched_reports),
        )
    trajectory = convert_sample_to_trajectory(
        sample,
        rollout_id=rollout_id,
        request_label=request_label,
        request_metadata=request_metadata,
        request_extra_fields=request_extra_fields,
        result_extra_fields=result.extra_fields,
        report=matched_report,
        default_model_name=report.model_name if report else None,
        unmatched_sample_reports=unmatched_reports or None,
    )
    dest = artifact_root / rollout_id / "trajectory.json"
    # Harbor's formatter keeps numeric arrays on one line.
    data = format_trajectory_json(trajectory.to_json_dict()).encode()
    await asyncio.to_thread(_write_document, dest, data)
    logger.info("Saved trajectory document for rollout %s -> %s", rollout_id, dest)
