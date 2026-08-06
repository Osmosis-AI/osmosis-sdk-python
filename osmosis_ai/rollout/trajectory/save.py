"""Best-effort saving of finished rollouts as ATIF trajectory documents.

Documents land at ``<artifact_root>/<rollout_id>/trajectory.json``, next to
the rollout's file artifacts; backend diagnostics land alongside as
``diagnostics.json``. Failures are logged and swallowed.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from harbor.utils.trajectory_utils import format_trajectory_json

from osmosis_ai.rollout.trajectory.converter import convert_sample_to_trajectory
from osmosis_ai.rollout.trajectory.report import SampleReport, TrajectoryReport
from osmosis_ai.rollout.types import ExecutionResult
from osmosis_ai.rollout.utils.file_artifacts import default_artifact_root

logger: logging.Logger = logging.getLogger(__name__)


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

    ``diagnostics`` overrides ``result.extra_fields`` for the sidecar.
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
    # Written before the sample-None early return so failures leave a record.
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
