"""Resolve the code bundle a backend needs for its agent and grader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.packaging import (
    BundleInfo,
    build_bundle,
    inspect_bundle,
    project_dir_for,
)
from osmosis_ai.rollout.backend.harbor.backend import ensure_import_path
from osmosis_ai.rollout.utils.imports import resolve_object


def resolve_backend_bundle(
    *,
    agent: str | type | None,
    grader: type | str | None,
    workflow_config: Any = None,
    grader_config: Any = None,
    code_dir: Path | None = None,
    bundle: Path | None = None,
    native: bool = False,
) -> BundleInfo | None:
    """Native agents need a bundle only when a grader is delivered;
    workflow agents always need one."""
    if bundle is not None:
        return inspect_bundle(Path(bundle))
    if native:
        if grader is None:
            return None
        anchor = resolve_object(grader)
    else:
        if agent is None:
            raise ValueError("pass agent (a native name or an AgentWorkflow)")
        anchor = resolve_object(agent)
    wheel = build_bundle(
        code_dir or project_dir_for(anchor),
        workflow=None if native else ensure_import_path(agent),
        grader=ensure_import_path(grader) if grader else None,
        workflow_config=ensure_import_path(workflow_config) if workflow_config else None,
        grader_config=ensure_import_path(grader_config) if grader_config else None,
    )
    return inspect_bundle(wheel)
