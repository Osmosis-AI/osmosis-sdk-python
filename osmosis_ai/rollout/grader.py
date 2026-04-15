from abc import ABC, abstractmethod
from typing import Any

from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.types import GraderConfig


class Grader(ABC):
    def __init__(self, config: GraderConfig | None = None):
        self.config = config

    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any:
        raise NotImplementedError
