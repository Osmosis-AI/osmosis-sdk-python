from abc import ABC, abstractmethod
from typing import Any

from osmosis_ai.rollout_v2.context import GraderContext
from osmosis_ai.rollout_v2.types import GraderConfig


class Grader(ABC):
    def __init__(self, config: GraderConfig | None = None):
        self.config = config

    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any:
        raise NotImplementedError
