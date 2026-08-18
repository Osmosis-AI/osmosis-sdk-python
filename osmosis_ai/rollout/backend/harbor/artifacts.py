"""Host-side artifact movement around a finished trial."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from osmosis_ai.rollout.backend.harbor.trial import TRIAL_NAME_PREFIX
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
    _is_link,
    _remove_link,
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


#: Harbor's own per-trial diagnostics, retained beside the canonical artifacts.
#: Kept out of ``artifacts/`` deliberately: that tree is enumerated wholesale and
#: rendered by the platform (§2.6), while these stay local-only (§2.7).
TRIAL_LOG_ENTRIES: tuple[str, ...] = ("trial.log", "exception.txt", "agent", "verifier")

#: Retained per step of a multi-step trial, relative to ``steps/<name>/``.
TRIAL_STEP_LOG_ENTRIES: tuple[str, ...] = ("agent", "verifier")

TRIAL_LOGS_DIRNAME = "logs"


def _trial_log_entries(trial_dir: Path) -> list[str]:
    """Trial-relative paths worth retaining, including per-step directories.

    A multi-step trial keeps ``agent/`` and ``verifier/`` at the root only as
    transient mount targets: their contents are relocated into
    ``steps/<name>/`` after each step and the empty roots are removed. Retaining
    only the root entries would leave a multi-step run with nothing but
    ``trial.log`` once the trial directory is cleaned up.
    """
    entries = list(TRIAL_LOG_ENTRIES)
    steps_dir = trial_dir / "steps"
    if steps_dir.is_dir() and not _is_link(steps_dir):
        for step in sorted(steps_dir.iterdir()):
            if not step.is_dir() or _is_link(step):
                continue
            entries.extend(
                f"steps/{step.name}/{name}" for name in TRIAL_STEP_LOG_ENTRIES
            )
    return entries


def _safe_trial_logs_dir(artifact_root: Path, rollout_id: str) -> Path:
    """Create the retention directory, refusing a symlinked component.

    Same discipline as ``copy_artifact_tree``: the root may intentionally be a
    mount or link, but every descendant is untrusted and is walked one component
    at a time so a planted link cannot redirect writes out of the tree.
    """
    current = artifact_root.absolute()
    current.mkdir(parents=True, exist_ok=True)
    for part in (rollout_id, TRIAL_LOGS_DIRNAME):
        current = current / part
        if _is_link(current):
            raise OSError(f"Refusing symlink in trial-log destination: {current}")
        current.mkdir(exist_ok=True)
    return current


def retain_trial_logs(trials_dir: Path, artifact_root: Path, rollout_id: str) -> bool:
    """Copy Harbor's per-trial logs to ``<artifact_root>/<id>/logs/`` (§4.3).

    ``cleanup_successful_trials`` removes the trial directory once its artifacts
    are relocated, which otherwise takes Harbor's own agent/verifier output with
    it -- exactly the material needed to explain a low reward. Runs strictly
    after credential scrubbing, so nothing unredacted is copied.

    Returns whether the attempt completed: ``False`` says the trial directory is
    still the only copy of these logs, so the caller must keep it. A trial with
    nothing to retain returns ``True``.
    """
    trial_dir = trials_dir / f"{TRIAL_NAME_PREFIX}{rollout_id}"
    if not trial_dir.is_dir():
        return True
    destination = artifact_root / rollout_id / TRIAL_LOGS_DIRNAME
    copied = 0
    try:
        for name in _trial_log_entries(trial_dir):
            source = trial_dir / name
            if source.is_dir():
                copied += copy_artifact_tree(
                    source,
                    destination / name,
                    destination_root=artifact_root,
                    replace_destination=True,
                )
            elif source.is_file() and "/" not in name:
                # Contents only, and never through a link: these files cross the
                # sandbox trust boundary just like artifacts do. Nested names are
                # directory entries; a file at one would bypass the checked walk
                # in _safe_trial_logs_dir, so it is left alone.
                if _is_link(source):
                    logger.warning("Skipping symlinked trial log %s", source)
                    continue
                target = _safe_trial_logs_dir(artifact_root, rollout_id) / name
                # copyfile opens the destination for writing, which would
                # follow a planted link and truncate its target.
                if _is_link(target):
                    _remove_link(target)
                shutil.copyfile(source, target)
                copied += 1
    except Exception:
        logger.warning(
            "Failed to retain trial logs for rollout %s (best-effort)",
            rollout_id,
            exc_info=True,
        )
        return False
    if copied:
        logger.info(
            "Retained %d Harbor trial log file(s) for rollout %s", copied, rollout_id
        )
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
