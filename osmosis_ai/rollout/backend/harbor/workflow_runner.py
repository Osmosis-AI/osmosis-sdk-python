from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path
from typing import Any

from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    HarborAgentWorkflowContext,
    RolloutContext,
)
from osmosis_ai.rollout.utils.imports import resolve_object


async def run_workflow(
    config: dict[str, Any],
    prompt: list[dict[str, Any]],
    *,
    logs_dir: Path | None = None,
    environment: Any = None,
) -> dict[str, Any]:
    workflow_cls = resolve_object(config["workflow"])
    workflow_config = (
        copy.deepcopy(resolve_object(config["workflow_config"]))
        if "workflow_config" in config
        else None
    )

    rollout_ctx = RolloutContext(
        chat_completions_url=config.get("chat_completions_url", ""),
        api_key=config.get("api_key"),
        rollout_id=config.get("rollout_id", ""),
    )

    if environment is None:
        ctx = AgentWorkflowContext(
            prompt=prompt,
            config=workflow_config,
        )
    else:
        ctx = HarborAgentWorkflowContext(
            prompt=prompt,
            config=workflow_config,
            environment=environment,
        )

    meta: dict[str, Any] = {"id": config.get("id", "")}

    try:
        workflow = workflow_cls(workflow_config)
        with rollout_ctx:
            await workflow.run(ctx)

        samples = await rollout_ctx.get_samples()
        samples_data = {sid: s.model_dump() for sid, s in samples.items()}

        if logs_dir is not None:
            (logs_dir / "samples.json").write_text(
                json.dumps(samples_data, default=str)
            )

        meta["status"] = "success"
        meta["samples"] = samples_data

    except Exception as e:
        traceback.print_exc()
        meta["status"] = "failure"
        meta["err_message"] = str(e)

    return meta
