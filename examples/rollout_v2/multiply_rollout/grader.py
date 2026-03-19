import re

from osmosis_ai.rollout_v2.context import GraderContext
from osmosis_ai.rollout_v2.grader import Grader
from osmosis_ai.rollout_v2.types import GraderConfig


class MultiplyGraderConfig(GraderConfig):
    name: str = "MultiplyGrader"
    description: str = "Grades multiplication rollouts"


multiply_grader_config = MultiplyGraderConfig()


def extract_solution(solution_str):
    solution = re.search(r"####\s*([-+]?\d*\.?\d+)", solution_str)
    if not solution:
        return None
    return solution.group(1)


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

    async def grade(self, ctx: GraderContext):
        rollout_samples = ctx.get_samples()

        if "multiply" in rollout_samples:
            content = rollout_samples["multiply"].messages[-1]["content"]
            if isinstance(content, list):
                content = next(
                    (block["text"] for block in content if "text" in block), ""
                )
            reward = self.compute_reward(content, ctx.label)
            ctx.set_sample_reward("multiply", reward)
            print(f"[MultiplyGrader] reward = {reward}")
        else:
            for sample_id, _ in rollout_samples.items():
                ctx.set_sample_reward(sample_id, 0)
                print(f"[MultiplyGrader] reward for {sample_id} = 0")
