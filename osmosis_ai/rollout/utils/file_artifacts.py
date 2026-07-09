"""Rollout file artifacts: user code writes under ``ctx.artifacts_dir``."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

# Harbor's auto-collected convention directory inside the sandbox.
HARBOR_ARTIFACTS_DIR = Path("/logs/artifacts")

CREATE_ATTEMPTS = 3
CREATE_BACKOFF_SECONDS = 0.5  # doubled per retry

_HEALTH_CHECK_FILENAME = ".osmosis-health-check"


def default_artifact_root() -> Path:
    """Host directory that holds one subdirectory per rollout id.

    Each rollout's outputs live at ``<root>/<rollout_id>/`` — file
    artifacts under ``artifacts/`` and the archived trajectory as
    ``trajectory.json`` — which the platform persists to durable storage.
    """
    return Path.home() / ".osmosis"


def _is_single_path_segment(name: str) -> bool:
    """True when ``name`` is a plain path component, not a traversal or root escape.

    ``rollout_id`` comes from the untrusted rollout request, so values like
    ``../other`` or ``/tmp/other`` must not be joined onto the artifact root or
    they'd let a rollout write outside its own directory.
    """
    if not name or name in (".", ".."):
        return False
    if os.sep in name or (os.altsep and os.altsep in name):
        return False
    return Path(name).name == name


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
    the rollout's artifacts.
    """
    if not _is_single_path_segment(rollout_id):
        logger.warning(
            "Refusing artifacts dir for rollout %r: id is not a single path "
            "segment (best-effort)",
            rollout_id,
        )
        return None
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
