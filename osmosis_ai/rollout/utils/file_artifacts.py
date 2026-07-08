"""Rollout file artifacts: user code writes under ``ctx.artifacts_dir``."""

from __future__ import annotations

from pathlib import Path

# Harbor's auto-collected convention directory inside the sandbox.
HARBOR_ARTIFACTS_DIR = Path("/logs/artifacts")

# Host subpath where Harbor lands it; LocalBackend reuses it to match.
HARBOR_COLLECTED_ARTIFACTS_SUBPATH = HARBOR_ARTIFACTS_DIR.relative_to("/")


def default_artifact_root() -> Path:
    """Host directory that holds one subdirectory per rollout id.

    Each rollout's outputs live at ``<root>/<rollout_id>/`` — file
    artifacts under ``artifacts/`` and the archived trajectory as
    ``trajectory.json`` — which the platform persists to durable storage.
    """
    return Path.home() / ".osmosis"
