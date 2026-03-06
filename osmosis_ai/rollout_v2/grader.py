from abc import ABC, abstractmethod
from typing import Any

from osmosis_ai.rollout_v2.context import GraderContext

class Grader(ABC):
    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any:
        raise NotImplementedError