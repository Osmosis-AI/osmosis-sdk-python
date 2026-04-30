"""Run an agent workflow inside a Harbor container.

Usage:
    osmosis-agent-runner --config /workspace/rollout_config.json --prompt /logs/agent/prompt.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.context import AgentWorkflowContext, RolloutContext
from osmosis_ai.rollout.utils.imports import resolve_object

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


async def run_workflow(
    config: dict[str, Any], prompt: list[dict[str, Any]]
) -> dict[str, Any]:
    workflow_cls = resolve_object(config["workflow"])
    workflow_config = (
        resolve_object(config["workflow_config"])
        if "workflow_config" in config
        else None
    )

    rollout_ctx = RolloutContext(
        chat_completions_url=config.get("chat_completions_url", ""),
        api_key=config.get("api_key"),
        rollout_id=config.get("rollout_id", ""),
    )

    ctx = AgentWorkflowContext(
        prompt=prompt,
        config=workflow_config,
    )

    meta: dict[str, Any] = {"id": config.get("id", "")}

    try:
        workflow = workflow_cls(workflow_config)
        with rollout_ctx:
            await workflow.run(ctx)

        samples = rollout_ctx.get_samples()
        samples_data = {sid: s.model_dump() for sid, s in samples.items()}

        (AGENT_LOGS_DIR / "samples.json").write_text(
            json.dumps(samples_data, default=str)
        )

        meta["status"] = "success"
        meta["samples"] = samples_data
        print(f"Agent runner complete: {len(samples)} samples collected")

    except Exception as e:
        traceback.print_exc()
        meta["status"] = "failure"
        meta["err_message"] = str(e)

    return meta


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompt = load_prompt(args.prompt)

    meta = asyncio.run(run_workflow(config, prompt))
    (AGENT_LOGS_DIR / "rollout_meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
