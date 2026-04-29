from typing import Any

from agents import ModelSettings, Runner
from agents.models.interface import Model

from multiply_rollout_openai.tools import multiply_tool
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.integrations.agents.openai_agents import (
    OsmosisAgent,
    OsmosisMemorySession,
    OsmosisRolloutModel,
)
from osmosis_ai.rollout.types import AgentWorkflowConfig


class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyAgentWorkflow"
    description: str = "Multiply two numbers"
    model: Model
    model_settings: ModelSettings
    tools: Any


multiply_workflow_config = MultiplyAgentWorkflowConfig(
    model=OsmosisRolloutModel(),
    model_settings=ModelSettings(temperature=1.0, top_p=1.0, max_tokens=4096),
    tools=[multiply_tool],
)


class MultiplyWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext):
        config = ctx.config

        agent = OsmosisAgent(
            name="multiply",
            model=config.model,
            model_settings=config.model_settings,
            tools=config.tools,
        )

        session = OsmosisMemorySession(name="multiply")
        await Runner.run(
            agent,
            ctx.prompt,
            session=session,
            max_turns=8,
        )
