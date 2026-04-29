from multiply_rollout_openai.utils import extract_solution
from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import GraderConfig


class MultiplyGraderConfig(GraderConfig):
    name: str = "MultiplyGrader"
    description: str = "Grades multiplication rollouts"


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

    async def grade(self, ctx: GraderContext):
        rollout_samples = ctx.get_samples()

        if "multiply" in rollout_samples:
            content = rollout_samples["multiply"].messages[-1].get("content", "")
            if isinstance(content, list):
                # Chat-completion content can be a list of parts; pick text parts.
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            reward = self.compute_reward(content, ctx.label)
            ctx.set_sample_reward("multiply", reward)
            print(f"[MultiplyGrader] reward = {reward}")
        else:
            for sample_id in rollout_samples:
                ctx.set_sample_reward(sample_id, 0)
                print(f"[MultiplyGrader] reward for {sample_id} = 0")
