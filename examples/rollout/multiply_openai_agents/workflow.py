from typing import Any

from agents import Agent

from multiply_openai_agents.tools import multiply_tool
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.integrations.agents.openai_agents import Runner
from osmosis_ai.rollout.types import AgentWorkflowConfig

MAX_TURNS = 8


class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyOpenAIAgentWorkflow"
    description: str = "Multiply two numbers (openai-agents variant)"
    tools: Any = None


multiply_workflow_config = MultiplyAgentWorkflowConfig(tools=[multiply_tool])


class MultiplyWorkflow(AgentWorkflow):
    """openai-agents variant of the multiply workflow.

    Unlike the Strands variant — which owns its own ``for _ in range(N)``
    loop and uses ``check_done`` to break early when the assistant stops
    emitting tool_calls — openai-agents' ``Runner`` owns the agent loop
    and already terminates automatically once the model returns a turn
    with no tool_calls (see ``tool_use_behavior="run_llm_again"``, the
    default). ``max_turns`` is the safety cap. All answer extraction and
    scoring lives in the grader.
    """

    async def run(self, ctx: AgentWorkflowContext) -> None:
        config = ctx.config
        agent = Agent(
            name="multiply",
            tools=config.tools if config else [multiply_tool],
        )
        await Runner.run(agent, ctx.prompt, max_turns=MAX_TURNS)
