from typing import Any

from strands.agent.agent_result import AgentResult
from strands.models.model import Model

from multiply_rollout.tools import multiply_tool
from multiply_rollout.utils import extract_solution
from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow
from osmosis_ai.rollout_v2.context import AgentWorkflowContext
from osmosis_ai.rollout_v2.integrations.agents.strands import (
    OsmosisRolloutModel,
)
from osmosis_ai.rollout_v2.integrations.agents.strands import (
    OsmosisStrandsAgent as StrandsAgent,
)
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig


class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyAgentWorkflow"
    description: str = "Multiply two numbers"
    model: Model
    tools: Any


multiply_workflow_config = MultiplyAgentWorkflowConfig(
    model=OsmosisRolloutModel(
        params={
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 4096,
        }
    ),
    tools=[multiply_tool],
)


class MultiplyWorkflow(AgentWorkflow):
    async def check_done(self, result: AgentResult) -> bool:
        message = result.message
        content = message.get("content", "")

        if not any("toolUse" in cb for cb in content):
            return True

        text_content = next((cb for cb in content if cb.get("text", None)), None)
        if text_content:
            solution = extract_solution(text_content.get("text", ""))
            if solution:
                return True

        return False

    async def run(self, ctx: AgentWorkflowContext):
        config = ctx.config

        agent = StrandsAgent(
            name="multiply",
            model=config.model,
            tools=config.tools,
            messages=ctx.prompt,
            callback_handler=None,
        )

        for _ in range(8):
            result = await agent.invoke_async()
            if await self.check_done(result):
                break
