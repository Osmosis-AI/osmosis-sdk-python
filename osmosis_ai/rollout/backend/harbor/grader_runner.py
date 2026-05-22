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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label = config.get("label")

    if not label:
        print("No label in config, skipping grading")
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

    ctx = GraderContext(label=label, sample=sample)
    grader = grader_cls(grader_config)
    asyncio.run(grader.grade(ctx))

    if ctx.sample is None or ctx.sample.reward is None:
        raise RuntimeError("Sample has no reward after grading")

    write_reward(ctx.sample.reward)
    print(f"Grading complete: reward={ctx.sample.reward}")


if __name__ == "__main__":
    main()
