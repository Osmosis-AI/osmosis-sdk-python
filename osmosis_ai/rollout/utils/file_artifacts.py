"""Rollout file artifacts: user code writes under ``ctx.artifacts_dir``."""

from __future__ import annotations

import asyncio
import logging
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
