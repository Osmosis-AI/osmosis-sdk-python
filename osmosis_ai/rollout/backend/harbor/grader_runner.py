"""Grade agent samples inside a Harbor container.

Usage:
    osmosis-grader-runner --config /workspace/rollout_config.json --samples /logs/agent/samples.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.imports import resolve_object

VERIFIER_LOGS_DIR = Path("/logs/verifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade agent samples")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to rollout_config.json"
    )
    parser.add_argument(
        "--samples", type=Path, required=True, help="Path to samples.json"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_samples(path: Path) -> dict[str, RolloutSample]:
    raw: dict[str, Any] = json.loads(path.read_text())
    return {sid: RolloutSample.model_validate(data) for sid, data in raw.items()}


def write_rewards(rewards: dict[str, float | None]) -> None:
    VERIFIER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOGS_DIR / "reward.json").write_text(json.dumps(rewards))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label = config.get("label")

    if not label:
        print("No label in config, skipping grading")
        write_rewards({})
        return

    if not args.samples.exists():
        print("No samples found, skipping grading")
        write_rewards({})
        return

    if "grader" not in config:
        print("No grader in config, skipping grading")
        write_rewards({})
        return

    samples = load_samples(args.samples)
    grader_cls = resolve_object(config["grader"])
    grader_config = (
        resolve_object(config["grader_config"]) if "grader_config" in config else None
    )

    ctx = GraderContext(label=label, samples=samples)
    grader = grader_cls(grader_config)
    asyncio.run(grader.grade(ctx))

    graded = ctx.get_samples()
    for sid, sample in graded.items():
        if sample.reward is None:
            raise RuntimeError(f"Sample {sid} has no reward after grading")

    rewards = {sid: sample.reward for sid, sample in graded.items()}
    write_rewards(rewards)
    print(f"Grading complete: {rewards}")


if __name__ == "__main__":
    main()
