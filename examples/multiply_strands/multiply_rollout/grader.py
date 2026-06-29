import logging
from typing import Any

from multiply_rollout.utils import extract_solution
from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.types import GraderConfig

logger = logging.getLogger(__name__)


class MultiplyGraderConfig(GraderConfig):
    name: str = "MultiplyStrandsGrader"
    description: str = "Grades multiplication rollouts using Strands"


multiply_grader_config = MultiplyGraderConfig()


class MultiplyGrader(Grader):
    def compute_reward(self, solution_str: str, ground_truth: str) -> float:
        extracted = extract_solution(solution_str)
        try:
            sol_val = float(extracted)
            gt_val = float(ground_truth)
        except (TypeError, ValueError):
            return 0.0

        if abs(gt_val - sol_val) < 1e-2:
            return 1.0
        return 0.0

    async def grade(self, ctx: GraderContext) -> None:
        if ctx.sample is None:
            ctx.set_reward(0.0)
            return

        last_message = ctx.sample.messages[-1]
        content = last_message.get("content", "")
        reward = self.compute_reward(_extract_text(content), ctx.label or "")
        ctx.set_reward(reward)
        logger.info("[MultiplyGrader] reward = %.1f", reward)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        return "\n".join(parts)
    return ""
