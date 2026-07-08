"""Best-effort saving of finished rollouts as ATIF trajectory documents.

Documents land at ``<artifact_root>/<rollout_id>/trajectory.json``, next to
the rollout's file artifacts. While the transitional multi-sample protocol
is still in use, a rollout with several samples writes one
``trajectory-<sample_id>.json`` per sample instead. Failures are logged
and swallowed.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from osmosis_ai.rollout._trajectory.converter import convert_sample_to_trajectory
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


async def save_trajectories(
    *,
    rollout_id: str,
    result: ExecutionResult,
    request_label: str | None = None,
    request_metadata: dict[str, Any] | None = None,
    request_extra_fields: dict[str, Any] | None = None,
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
    artifact_root: Path,
) -> None:
    dest_dir = artifact_root / rollout_id
    written = 0
    single = len(result.samples) == 1
    for sample_id, sample in result.samples.items():
        try:
            trajectory = convert_sample_to_trajectory(
                sample,
                rollout_id=rollout_id,
                sample_id=sample_id,
                request_label=request_label,
                request_metadata=request_metadata,
                request_extra_fields=request_extra_fields,
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
        data = json.dumps(
            trajectory.to_json_dict(), ensure_ascii=False, indent=2
        ).encode()
        await asyncio.to_thread(_write_document, dest_dir / name, data)
        written += 1

    if written:
        logger.info(
            "Saved %d trajectory document(s) for rollout %s -> %s",
            written,
            rollout_id,
            dest_dir,
        )
