"""<your-rollout>: placeholder rollout server created by `osmosis rollout init`.

Fill in two methods, then run `python main.py` to start a FastAPI rollout
server on $ROLLOUT_PORT (default 8000):

  - MyAgentWorkflow.run():  drive the LLM and register sample sources.
  - MyGrader.grade():       turn samples into scalar rewards.

Compare the multiply-* rollouts in the workspace-template repo for fully
worked examples (Strands, OpenAI Agents, Harbor-backed).
"""

from __future__ import annotations

import os

import uvicorn

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local import LocalBackend
from osmosis_ai.rollout.context import AgentWorkflowContext, GraderContext
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.server import create_rollout_server


class MyAgentWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> None:
        # TODO: implement agent logic.
        #
        # ctx.prompt is the list[dict] of input messages for this rollout.
        # Drive the LLM with one of:
        #   * osmosis_ai.rollout.integrations.agents.strands (Strands)
        #   * osmosis_ai.rollout.integrations.agents.openai_agents (OpenAI Agents SDK)
        #   * raw HTTP against get_rollout_context().chat_completions_url
        #
        # Register a SampleSource so the grader can read the conversation:
        #   from osmosis_ai.rollout.context import get_rollout_context
        #   get_rollout_context().register_sample_source("my-agent", source)
        raise NotImplementedError(
            f"MyAgentWorkflow.run() received {len(ctx.prompt)} prompt message(s) "
            "but the workflow body is not implemented yet."
        )


class MyGrader(Grader):
    async def grade(self, ctx: GraderContext) -> None:
        # TODO: implement grading.
        #
        # ctx.label is the dataset row's ground truth (string).
        # ctx.get_samples() returns {name: RolloutSample} for sources registered
        # by the workflow. Assign a scalar reward per sample with:
        #   ctx.set_sample_reward(name, value)
        for sample_id in ctx.get_samples():
            ctx.set_sample_reward(sample_id, 0.0)


def main() -> None:
    backend = LocalBackend(workflow=MyAgentWorkflow, grader=MyGrader)
    app = create_rollout_server(backend=backend)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ROLLOUT_PORT", "8000")))


if __name__ == "__main__":
    main()
