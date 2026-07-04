"""Rollout file artifacts: user code writes under ``ctx.artifacts_dir``;
backends land the files at ``<artifact_root>/<rollout_id>/artifacts/``."""

from __future__ import annotations

from pathlib import Path

# Harbor's agent publish dir inside the sandbox.
HARBOR_ARTIFACTS_DIR = Path("/logs/artifacts")


def default_artifact_root() -> Path:
    """Host directory that holds one subdirectory per rollout id."""
    return Path.home() / ".osmosis" / "artifacts"
