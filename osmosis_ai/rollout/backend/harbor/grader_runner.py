"""Grade the rollout's single sample inside a Harbor container.

Usage:
    osmosis-grader-runner --config /workspace/rollout_config.json --sample /logs/agent/sample.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.file_artifacts import (
    GRADER_ARTIFACTS_SNAPSHOT_DIRNAME,
    HARBOR_ARTIFACTS_DIR,
    ArtifactFileState,
    artifact_tree_state,
    copy_artifact_tree,
)
from osmosis_ai.rollout.utils.imports import resolve_object

VERIFIER_LOGS_DIR = Path("/logs/verifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade the rollout sample")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to rollout_config.json"
    )
    parser.add_argument(
        "--sample", type=Path, required=True, help="Path to sample.json"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_sample(path: Path) -> RolloutSample | None:
    raw: Any = json.loads(path.read_text())
    if raw is None:
        return None
    return RolloutSample.model_validate(raw)


def write_reward(reward: float | None) -> None:
    VERIFIER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOGS_DIR / "reward.json").write_text(json.dumps({"reward": reward}))


def capture_artifact_baseline() -> ArtifactFileState:
    """Capture pre-grader files without making artifact I/O grading-critical."""
    try:
        return artifact_tree_state(HARBOR_ARTIFACTS_DIR)
    except Exception as exc:
        print(
            f"Failed to inspect pre-grader artifacts (best-effort): {exc}",
            file=sys.stderr,
        )
        return {}


def stage_grader_artifacts(baseline: ArtifactFileState) -> None:
    """Stage the post-grader artifact tree in Harbor's returned verifier logs.

    A snapshot directory exists afterwards only when the grader changed files:
    stale snapshots are removed first, and an empty copy result is dropped so
    the host backend can treat the snapshot's presence as "there is an increment".
    """
    snapshot_dir = VERIFIER_LOGS_DIR / GRADER_ARTIFACTS_SNAPSHOT_DIRNAME
    try:
        if snapshot_dir.is_junction():
            snapshot_dir.rmdir()
        elif snapshot_dir.is_symlink() or (
            snapshot_dir.exists() and not snapshot_dir.is_dir()
        ):
            snapshot_dir.unlink()
        elif snapshot_dir.is_dir():
            shutil.rmtree(snapshot_dir)
        if not HARBOR_ARTIFACTS_DIR.is_dir():
            return
        copied = copy_artifact_tree(
            HARBOR_ARTIFACTS_DIR,
            snapshot_dir,
            destination_root=VERIFIER_LOGS_DIR,
            baseline=baseline,
        )
        if not copied:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
    except Exception as exc:
        # File artifacts are best-effort and must never mask a grader result.
        print(f"Failed to stage grader artifacts (best-effort): {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label = config.get("label")
    metadata = config.get("metadata")

    if not label and metadata is None:
        print("No label or metadata in config, skipping grading")
        write_reward(None)
        return

    if not args.sample.exists():
        print("No sample found, skipping grading")
        write_reward(None)
        return

    if "grader" not in config:
        print("No grader in config, skipping grading")
        write_reward(None)
        return

    sample = load_sample(args.sample)
    if sample is None:
        print("Sample file is empty, skipping grading")
        write_reward(None)
        return

    artifact_baseline = capture_artifact_baseline()
    try:
        # Resolving user modules and constructing the grader are part of its
        # lifecycle: either step may write diagnostics before failing.
        grader_cls = resolve_object(config["grader"])
        grader_config = (
            resolve_object(config["grader_config"])
            if "grader_config" in config
            else None
        )
        ctx = GraderContext(
            label=label,
            sample=sample,
            metadata=metadata,
            artifacts_dir=HARBOR_ARTIFACTS_DIR,
        )
        grader = grader_cls(grader_config)
        asyncio.run(grader.grade(ctx))
    finally:
        # Harbor 0.16 collects /logs/artifacts before running the verifier. Its
        # verifier directory, however, is returned after verification for both
        # shared and separate environments. Snapshot here so grader-authored files
        # survive until HarborBackend.on_trial_end can merge and relocate them.
        stage_grader_artifacts(artifact_baseline)

    if ctx.sample is None or ctx.sample.reward is None:
        raise RuntimeError("Sample has no reward after grading")

    write_reward(ctx.sample.reward)
    print(f"Grading complete: reward={ctx.sample.reward}")


if __name__ == "__main__":
    main()
