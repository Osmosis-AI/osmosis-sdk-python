from typing import Any

from strands.models.model import Model

from multiply_rollout.tools import multiply_tool
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.integrations.agents.strands import (
    OsmosisRolloutModel,
    OsmosisStrandsAgent,
)
from osmosis_ai.rollout.types import AgentWorkflowConfig

MAX_ITERATIONS = 8


class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyAgentWorkflow"
    description: str = "Multiply two numbers using Strands"
    model: Model
    tools: Any


multiply_workflow_config = MultiplyAgentWorkflowConfig(
    model=OsmosisRolloutModel(params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096}),
    tools=[multiply_tool],
)


class MultiplyWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext) -> None:
        config = ctx.config
        system_prompt, user_prompt = _split_prompt(ctx.prompt)
        agent = OsmosisStrandsAgent(
            name="multiply",
            model=config.model,
            tools=config.tools,
            system_prompt=system_prompt,
        )
        await agent.invoke_async(user_prompt)


def _split_prompt(prompt: list[dict[str, Any]]) -> tuple[str | None, str]:
    system: str | None = None
    user_parts: list[str] = []
    for msg in prompt:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system" and isinstance(content, str):
            system = content
        elif role == "user" and isinstance(content, str):
            user_parts.append(content)
    return system, "\n".join(user_parts)
