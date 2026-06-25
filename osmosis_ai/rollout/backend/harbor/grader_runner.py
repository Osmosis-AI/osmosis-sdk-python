"""Grade the rollout's single sample inside a Harbor container.

Usage:
    osmosis-grader-runner --config /workspace/rollout_config.json --sample /logs/agent/sample.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.artifacts import sanitize_artifacts
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


def write_artifacts(artifacts: dict[str, Any]) -> None:
    """Write grader artifacts to the verifier dir (sibling of reward.json).

    Sanitized to mirror the server callback path, so a non-serializable or
    oversized payload becomes an ``_error`` marker rather than crashing the
    runner after rewards are already written.
    """
    sanitized = sanitize_artifacts(artifacts)
    if sanitized is None:
        return
    VERIFIER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Mirror sanitize_artifacts' compact UTF-8 settings so the on-disk file stays
    # within the size cap and matches the callback wire format.
    (VERIFIER_LOGS_DIR / "grader_artifacts.json").write_text(
        json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


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

    grader_cls = resolve_object(config["grader"])
    grader_config = (
        resolve_object(config["grader_config"]) if "grader_config" in config else None
    )

    ctx = GraderContext(label=label, sample=sample, metadata=metadata)
    grader = grader_cls(grader_config)
    asyncio.run(grader.grade(ctx))

    if ctx.sample is None or ctx.sample.reward is None:
        raise RuntimeError("Sample has no reward after grading")

    write_reward(ctx.sample.reward)
    if ctx.artifacts is not None:
        # Artifacts are best-effort and must never block reward delivery, even
        # if the verifier dir is unwritable after reward.json is persisted.
        try:
            write_artifacts(ctx.artifacts)
        except OSError as e:
            print(f"Warning: failed to write grader artifacts: {e}")
    print(f"Grading complete: reward={ctx.sample.reward}")


if __name__ == "__main__":
    main()
