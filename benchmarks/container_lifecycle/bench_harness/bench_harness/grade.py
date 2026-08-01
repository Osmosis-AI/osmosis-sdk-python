"""Bench grader: constant reward; exists to exercise the in-env grader path."""

from osmosis_ai.rollout import Grader, GraderContext


class BenchGrader(Grader):
    async def grade(self, ctx: GraderContext) -> None:
        ctx.set_reward(1.0)
