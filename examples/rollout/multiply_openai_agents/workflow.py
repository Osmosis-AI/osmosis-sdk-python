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
    model: Any = None
    tools: Any = None


multiply_workflow_config = MultiplyAgentWorkflowConfig(tools=[multiply_tool])


class MultiplyWorkflow(AgentWorkflow):
    """openai-agents variant of the multiply workflow.

    Training runs should leave ``model`` unset so the agent uses the rollout
    controller model supplied by the training backend. Set ``config.model`` only
    for ablations with another OpenAI Agents model while keeping sample
    recording.

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
            model=config.model if config else None,
            tools=config.tools if config else [multiply_tool],
        )
        # Training workflows should use run() with model unset. The wrapper
        # handles the controller SSE path and records the sample before returning.
        await Runner.run(agent, ctx.prompt, max_turns=MAX_TURNS)
