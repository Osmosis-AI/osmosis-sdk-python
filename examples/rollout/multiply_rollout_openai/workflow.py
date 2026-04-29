from typing import Any

from agents import Agent, Runner

from multiply_rollout_openai.tools import multiply_tool
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.integrations.agents.openai_agents import (
    OsmosisRolloutModel,
    OsmosisSession,
)
from osmosis_ai.rollout.types import AgentWorkflowConfig


class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyAgentWorkflow"
    description: str = "Multiply two numbers"
    instructions: str
    tools: Any


multiply_workflow_config = MultiplyAgentWorkflowConfig(
    instructions=(
        "You are a math assistant. Use the multiply tool to compute products. "
        "When you have the answer, respond with `#### <number>` on its own line."
    ),
    tools=[multiply_tool],
)


class MultiplyWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext):
        config = ctx.config

        agent = Agent(
            name="multiply",
            instructions=config.instructions,
            model=OsmosisRolloutModel(),
            tools=config.tools,
        )

        session = OsmosisSession(name="multiply")
        await Runner.run(
            agent,
            input=ctx.prompt,
            session=session,
            max_turns=8,
        )
