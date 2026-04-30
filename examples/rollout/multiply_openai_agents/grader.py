from typing import Any

from multiply_openai_agents.utils import extract_solution
from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import GraderConfig


class MultiplyGraderConfig(GraderConfig):
    name: str = "MultiplyOpenAIAgentsGrader"
    description: str = "Grades multiplication rollouts (openai-agents variant)"


multiply_grader_config = MultiplyGraderConfig()


class MultiplyGrader(Grader):
    def compute_reward(self, solution_str: str, ground_truth: str):
        extracted = extract_solution(solution_str)
        try:
            sol_val = float(extracted)
        except Exception:
            return 0.0

        gt_val = float(ground_truth)

        if abs(gt_val - sol_val) < 1e-2:
            return 1.0
        return 0.0

    async def grade(self, ctx: GraderContext) -> None:
        samples = ctx.get_samples()
        if "multiply" not in samples:
            for sample_id in samples:
                ctx.set_sample_reward(sample_id, 0.0)
                print(f"[MultiplyGrader] reward for {sample_id} = 0")
            return

        # openai-agents' ``to_input_list()`` emits Responses API items.
        # The final assistant message stores text under
        # ``content=[{"type": "output_text", "text": "..."}]``, unlike
        # chat-completions which uses a plain string.
        last_message = samples["multiply"].messages[-1]
        content = last_message.get("content", "")
        reward = self.compute_reward(_extract_text(content), ctx.label or "")
        ctx.set_sample_reward("multiply", reward)
        print(f"[MultiplyGrader] reward = {reward}")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "output_text"
        ]
        return "\n".join(parts)
    return ""
