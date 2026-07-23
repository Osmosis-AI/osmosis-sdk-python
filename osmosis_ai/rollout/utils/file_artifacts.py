"""Rollout file artifacts: user code writes under ``ctx.artifacts_dir``."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

# Harbor's auto-collected convention directory inside the sandbox.
HARBOR_ARTIFACTS_DIR = Path("/logs/artifacts")
# Harbor always returns /logs/verifier after the verifier exits. The grader runner
# stages its final artifacts tree under this reserved child so the host backend can
# merge grader-authored files into Harbor's earlier, pre-verifier artifact snapshot.
GRADER_ARTIFACTS_SNAPSHOT_DIRNAME = ".osmosis-artifacts"

CREATE_ATTEMPTS = 3
CREATE_BACKOFF_SECONDS = 0.5  # doubled per retry

_HEALTH_CHECK_FILENAME = ".osmosis-health-check"

type ArtifactFileState = dict[str, tuple[int, int, int]]


def artifact_tree_state(source: Path) -> ArtifactFileState:
    """Return cheap change-detection metadata for regular files in a tree."""
    state: ArtifactFileState = {}

    def _visit(directory: Path, relative_dir: Path) -> None:
        if directory.is_symlink() or not directory.is_dir():
            return
        for entry in sorted(directory.iterdir(), key=lambda path: path.name):
            if entry.is_symlink():
                continue
            relative_path = relative_dir / entry.name
            if entry.is_dir():
                _visit(entry, relative_path)
            elif entry.is_file():
                stat = entry.stat()
                state[relative_path.as_posix()] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )

    _visit(source, Path())
    return state


def copy_artifact_tree(
    source: Path,
    destination: Path,
    *,
    baseline: ArtifactFileState | None = None,
) -> int:
    """Merge regular files from ``source`` into ``destination``.

    Artifact trees cross trust and filesystem boundaries, so copy file contents
    only: never follow or recreate symlinks, special files, or metadata. Existing
    destination entries are replaced when their file/directory type changed. When
    ``baseline`` is provided, unchanged files are skipped. The merge is additive:
    files present only in ``destination`` are kept, so deletions in ``source``
    never propagate. Returns the number of files copied.
    """
    if source.is_symlink() or not source.is_dir():
        return 0

    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)

    def _copy(directory: Path, target_dir: Path, relative_dir: Path) -> int:
        copied = 0
        for entry in sorted(directory.iterdir(), key=lambda path: path.name):
            target = target_dir / entry.name
            relative_path = relative_dir / entry.name
            if entry.is_symlink():
                logger.warning("Skipping symlink in artifact tree: %s", entry)
                continue
            if entry.is_dir():
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    target.unlink()
                target.mkdir(parents=True, exist_ok=True)
                copied += _copy(entry, target, relative_path)
                continue
            if entry.is_file():
                stat = entry.stat()
                current_state = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if (
                    baseline is not None
                    and baseline.get(relative_path.as_posix()) == current_state
                ):
                    continue
                if target.is_symlink():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry, target)
                copied += 1
                continue
            logger.warning("Skipping special file in artifact tree: %s", entry)
        return copied

    return _copy(source, destination, Path())


def default_artifact_root() -> Path:
    """Host directory that holds one subdirectory per rollout id.

    Each rollout's outputs live at ``<root>/<rollout_id>/`` — file
    artifacts under ``artifacts/`` and the archived trajectory as
    ``trajectory.json`` — which the platform persists to durable storage.
    """
    return Path.home() / ".osmosis"


def _ensure_artifacts_dir(rollout_dir: Path) -> Path:
    artifacts_dir = rollout_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Create a marker to confirm the dir is writable; only once per rollout, so
    # a later call can't fail on a store that rejects overwrites.
    health_check = rollout_dir / _HEALTH_CHECK_FILENAME
    if not health_check.exists():
        health_check.write_bytes(b"osmosis-health-check")
    return artifacts_dir


async def create_rollout_artifacts_dir(
    root: Path, rollout_id: str, *, attempts: int | None = None
) -> Path | None:
    """Create and verify the rollout's artifacts dir; ``None`` when unusable.

    ``mkdir`` alone can succeed without proving the directory is writable, so the
    first call writes a check file to confirm it before user code relies on the
    dir. Retries ride out transient errors while the dir becomes available.

    The check file is written outside ``artifacts/`` so it doesn't appear among
    the rollout's artifacts. ``rollout_id`` must already be validated as a
    single path segment (see ``osmosis_ai.rollout.utils.identifiers``).
    """
    rollout_dir = root / rollout_id
    total = attempts if attempts is not None else CREATE_ATTEMPTS
    last_error: OSError | None = None
    for attempt in range(total):
        try:
            # Filesystem I/O may block; run it off the event loop.
            return await asyncio.to_thread(_ensure_artifacts_dir, rollout_dir)
        except OSError as error:
            last_error = error
            if attempt < total - 1:
                await asyncio.sleep(CREATE_BACKOFF_SECONDS * (2**attempt))
    logger.warning(
        "Artifacts dir unusable for rollout %s under %s (best-effort)",
        rollout_id,
        root,
        exc_info=last_error,
    )
    return None
