"""Host-side artifact movement around a finished trial."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from osmosis_ai.rollout.backend.harbor.backend import TRIAL_NAME_PREFIX
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
    copy_artifact_tree,
)

logger: logging.Logger = logging.getLogger(__name__)


def relocate_trial_artifacts(
    trials_dir: Path, artifact_root: Path, rollout_id: str, *, move: bool
) -> bool:
    source_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}" / "artifacts"
    if not source_dir.is_dir():
        return True
    try:
        copy_artifact_tree(
            source_dir,
            artifact_root / rollout_id / "artifacts",
            destination_root=artifact_root,
            replace_destination=True,
        )
        if move:
            shutil.rmtree(source_dir)
    except Exception:
        logger.warning(
            "Failed to relocate trial artifacts for rollout %s (best-effort)",
            rollout_id,
            exc_info=True,
        )
        return False
    return True


def merge_grader_artifacts(trials_dir: Path, rollout_id: str) -> None:
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
    source_dir = trial_dir / "verifier" / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
    if not source_dir.is_dir():
        return
    try:
        copy_artifact_tree(
            source_dir,
            trial_dir / "artifacts" / HARBOR_ARTIFACTS_DIR.relative_to("/"),
            destination_root=trials_dir,
        )
    except Exception:
        logger.warning(
            "Failed to merge grader artifacts for rollout %s (best-effort)",
            rollout_id,
            exc_info=True,
        )
