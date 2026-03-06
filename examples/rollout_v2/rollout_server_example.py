import uvicorn
import logging
import re
from typing import Any

from rich import print

from strands.models.model import Model
from strands import tool
from strands.agent.agent_result import AgentResult

from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.context import GraderContext

from osmosis_ai.rollout_v2.rollout_server_base import create_app
from osmosis_ai.rollout_v2.agent_workflow import AgentWorkflow, AgentWorkflowContext
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig
from osmosis_ai.rollout_v2.integrations.strands import OsmosisStrandsAgent as StrandsAgent, OsmosisRolloutModel

# import litellm
# litellm._turn_on_debug()

# # Configure the root strands logger
# logging.getLogger("strands").setLevel(logging.DEBUG)

# # Add a handler to see the logs
# logging.basicConfig(
#     format="%(levelname)s | %(name)s | %(message)s",
#     handlers=[logging.StreamHandler()]
# )

logger = logging.getLogger(__name__)

def extract_solution(solution_str):
    solution = re.search(r'####\s*([-+]?\d*\.?\d+)', solution_str)
    if(not solution or solution is None):
        return None
    final_solution = solution.group(1)
    return final_solution

class MultiplyGrader(Grader):
    def compute_reward(self, solution_str: str, ground_truth: str, extra_info: dict=None, **kwargs):
        extracted = extract_solution(solution_str)
        try:
            sol_val = float(extracted)
        except:
            return 0.0

        gt_val = float(ground_truth)

        if(sol_val is None):
            return 0.0

        if(abs(gt_val - sol_val) < 1e-2):
            return 1.0
        return 0.0

    async def grade(self, ctx: GraderContext):
        rollout_samples = ctx.get_samples()

        if "multiply" in rollout_samples:
            reward = self.compute_reward(rollout_samples["multiply"].messages[-1]["content"], rollout_samples["multiply"].label)
            ctx.set_sample_reward("multiply", reward)
        else:
            for sample_id, _ in rollout_samples.items():
                ctx.set_sample_reward(sample_id, 0)

@tool(name="multiply")
def multiply_tool(a: float, b: float) -> float:
    """
    Multiply two numbers

    # Args:
    - a: The first number
    - b: The second number

    # Returns:
    - The product of the two numbers
    """
    return a * b

class MultiplyAgentWorkflow(AgentWorkflow):
    async def check_done(self, result: AgentResult) -> bool:
        message = result.message
        content = message.get("content", "")

        # if no items in the content are toolUse, return True
        if not any("toolUse" in cb for cb in content):
            return True

        # If a solution is outputted, end the conversation
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

        num_turns = 8

        for _ in range(num_turns):
            print(agent.messages)
            result = await agent.invoke_async()
            print(result)

            if await self.check_done(result):
                break

class MultiplyAgentWorkflowConfig(AgentWorkflowConfig):
    name: str = "MultiplyAgentWorkflow"
    description: str = "Multiply two numbers"
    model: Model
    tools: Any

def multiply():
    config = MultiplyAgentWorkflowConfig(
        name="MultiplyAgentWorkflow",
        description="Multiply two numbers",
        tools=[multiply_tool],
        model=OsmosisRolloutModel(
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 4096,
            }
        )
    )
    app = create_app(agent_workflow_cls=MultiplyAgentWorkflow, grader_cls=MultiplyGrader, agent_workflow_config=config)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    multiply()