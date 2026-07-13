"""Best-effort saving of finished rollouts as ATIF trajectory documents.

Documents land at ``<artifact_root>/<rollout_id>/trajectory.json``, next to
the rollout's file artifacts. While the transitional multi-sample protocol
is still in use, a rollout with several samples writes one
``trajectory-<sample_id>.json`` per sample instead. Failures are logged
and swallowed.
"""

import asyncio
import logging
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from harbor.utils.trajectory_utils import format_trajectory_json

from osmosis_ai.rollout.trajectory.converter import convert_sample_to_trajectory
from osmosis_ai.rollout.trajectory.report import SampleReport, TrajectoryReport
from osmosis_ai.rollout.types import ExecutionResult
from osmosis_ai.rollout.utils.file_artifacts import default_artifact_root

logger: logging.Logger = logging.getLogger(__name__)

_SEGMENT_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(value: str) -> str:
    cleaned = _SEGMENT_SANITIZER.sub("_", value).lstrip(".")
    return cleaned or "unnamed"


def _write_document(dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def _resolve_sample_reports(
    report: TrajectoryReport | None, sample_ids: Collection[str]
) -> tuple[dict[str, SampleReport], dict[str, SampleReport]]:
    """Match report entries to samples by id; return ``(matched, unmatched)``.

    A lone entry for a lone sample matches regardless of key, for
    controllers that cannot know the server-side sample ids.
    """
    if report is None or not report.samples:
        return {}, {}
    matched = {
        sample_id: report.samples[sample_id]
        for sample_id in sample_ids
        if sample_id in report.samples
    }
    unmatched = {
        key: value for key, value in report.samples.items() if key not in matched
    }
    if not matched and len(unmatched) == 1 and len(sample_ids) == 1:
        (sample_id,) = sample_ids
        key, value = next(iter(unmatched.items()))
        logger.debug(
            "Trajectory report entry %r applied to the rollout's only sample %r",
            key,
            sample_id,
        )
        return {sample_id: value}, {}
    return matched, unmatched


async def save_trajectories(
    *,
    rollout_id: str,
    result: ExecutionResult,
    request_label: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    request_extra_fields: dict[str, Any] | None = None,
    report: TrajectoryReport | None = None,
    artifact_root: Path | None = None,
) -> None:
    """Save one rollout's samples as ATIF documents. Never raises."""
    try:
        await _save(
            rollout_id=rollout_id,
            result=result,
            request_label=request_label,
            request_metadata=request_metadata,
            request_extra_fields=request_extra_fields,
            report=report,
            artifact_root=artifact_root or default_artifact_root(),
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
) -> None:
    dest_dir = artifact_root / rollout_id
    written = 0
    single = len(result.samples) == 1
    matched_reports, unmatched_reports = _resolve_sample_reports(
        report, result.samples.keys()
    )
    if unmatched_reports:
        logger.warning(
            "Trajectory report for rollout %s has entries for unknown sample "
            "ids %s (samples: %s); %s",
            rollout_id,
            sorted(unmatched_reports),
            sorted(result.samples),
            "preserving them under extra.osmosis.unmatched_sample_reports"
            if single
            else "dropping them",
        )
    for sample_id, sample in result.samples.items():
        if sample.trajectory_messages is None:
            # Explicit opt-out, or an upstream conversion/snapshot failure
            # that already warned with a traceback -- not worth a warning here.
            logger.debug(
                "Skipping trajectory for sample %s of rollout %s: no trajectory "
                "messages (persistence disabled or conversion failed upstream)",
                sample_id,
                rollout_id,
            )
            continue
        try:
            trajectory = convert_sample_to_trajectory(
                sample,
                rollout_id=rollout_id,
                sample_id=sample_id,
                request_label=request_label,
                request_metadata=request_metadata,
                request_extra_fields=request_extra_fields,
                report=matched_reports.get(sample_id),
                default_model_name=report.model_name if report else None,
                unmatched_sample_reports=unmatched_reports if single else None,
            )
        except Exception:
            logger.warning(
                "Failed to convert sample %s of rollout %s to ATIF (best-effort)",
                sample_id,
                rollout_id,
                exc_info=True,
            )
            continue
        name = (
            "trajectory.json"
            if single
            else f"trajectory-{_safe_segment(sample_id)}.json"
        )
        try:
            # Harbor's formatter keeps numeric arrays on one line.
            data = format_trajectory_json(trajectory.to_json_dict()).encode()
            await asyncio.to_thread(_write_document, dest_dir / name, data)
        except Exception:
            logger.warning(
                "Failed to write trajectory document for sample %s of rollout %s "
                "(best-effort)",
                sample_id,
                rollout_id,
                exc_info=True,
            )
            continue
        written += 1

    if written:
        logger.info(
            "Saved %d trajectory document(s) for rollout %s -> %s",
            written,
            rollout_id,
            dest_dir,
        )
