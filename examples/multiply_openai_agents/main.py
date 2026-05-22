"""Multiply rollout server: minimal end-to-end Osmosis rollout example.

Self-contained FastAPI rollout server that drives a small OpenAI Agents SDK
workflow against the active rollout's chat-completions endpoint. The agent
has one tool (``multiply``) and a numeric grader that scores the final
answer against the dataset label.

Bring up the multi-LoRA + remote-rollout cluster (see
``osmosis-traingate/specs/experiment/miles-multi-lora-service.toml``), then
point an adapter at this server via ``metadata.rollout_server_url`` and the
trainer's custom generate function will POST rollouts here.

Standalone boot (sanity check):

    pip install -e '../..[server]' 'openai-agents>=0.14,<0.15' uvicorn
    python main.py &
    curl http://localhost:8080/health
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import uvicorn
from agents import ModelSettings, Runner, function_tool

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local import LocalBackend
from osmosis_ai.rollout.context import AgentWorkflowContext, GraderContext
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.integrations.agents.openai_agents import (
    OsmosisAgent,
    OsmosisMemorySession,
    OsmosisRolloutModel,
)
from osmosis_ai.rollout.server import create_rollout_server

logger = logging.getLogger(__name__)

MAX_TURNS = 8
# Loose tolerance so float-y answers like "12.0" match integer labels.
ANSWER_TOLERANCE = 1e-2


@function_tool
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


class MultiplyWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> None:
        agent = OsmosisAgent(
            name="multiply",
            instructions=(
                "Solve the user's multiplication problem. You may call the "
                "`multiply` tool. End your final message with "
                "`Answer: <number>` on its own line."
            ),
            model=OsmosisRolloutModel(),
            model_settings=ModelSettings(temperature=1.0, max_tokens=512),
            tools=[multiply],
        )
        session = OsmosisMemorySession()
        await Runner.run(agent, ctx.prompt, session=session, max_turns=MAX_TURNS)


_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _extract_text(content: Any) -> str:
    """Flatten OpenAI Responses-API content into plain text.

    ``content`` can be a str, or a list of typed parts; we keep the text-ish
    parts (``text`` / ``output_text``) and concatenate.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in ("text", "output_text")
        )
    return ""


class MultiplyGrader(Grader):
    """0/1 reward: match the trailing ``Answer: <n>`` against ``ctx.label``."""

    async def grade(self, ctx: GraderContext) -> None:
        if ctx.sample is None or ctx.label is None:
            ctx.set_reward(0.0)
            return

        # Walk back to the most recent assistant turn with text content;
        # skips tool-call / tool-result turns that have no human-readable body.
        text = ""
        for msg in reversed(ctx.sample.messages):
            if msg.get("role") != "assistant":
                continue
            text = _extract_text(msg.get("content", ""))
            if text:
                break

        match = _ANSWER_RE.search(text)
        if match is None:
            ctx.set_reward(0.0)
            return

        try:
            predicted = float(match.group(1))
            target = float(ctx.label)
        except ValueError:
            ctx.set_reward(0.0)
            return

        reward = 1.0 if abs(predicted - target) < ANSWER_TOLERANCE else 0.0
        ctx.set_reward(reward)
        logger.info(
            "multiply reward=%.1f predicted=%s target=%s",
            reward, predicted, target,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[multiply-rollout] %(message)s"
    )
    backend = LocalBackend(workflow=MultiplyWorkflow, grader=MultiplyGrader)
    app = create_rollout_server(backend=backend)
    port = int(os.environ.get("ROLLOUT_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
