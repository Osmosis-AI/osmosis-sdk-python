"""Run an agent workflow inside a Harbor container.

Usage:
    osmosis-agent-runner --config /workspace/rollout_config.json --prompt /logs/agent/prompt.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.backend.harbor.workflow_runner import run_workflow

AGENT_LOGS_DIR = Path("/logs/agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent workflow")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to rollout_config.json"
    )
    parser.add_argument(
        "--prompt", type=Path, required=True, help="Path to prompt.json"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_prompt(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompt = load_prompt(args.prompt)

    meta = asyncio.run(run_workflow(config, prompt, logs_dir=AGENT_LOGS_DIR))
    if meta.get("status") == "success":
        print(
            f"Agent runner complete: {len(meta.get('samples', {}))} samples collected"
        )
    (AGENT_LOGS_DIR / "rollout_meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
