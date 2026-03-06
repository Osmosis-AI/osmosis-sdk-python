from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from osmosis_ai.rollout_v2.types import AgentWorkflowConfig
from osmosis_ai.rollout_v2.context import AgentWorkflowContext

TConfig = TypeVar("TConfig", bound=AgentWorkflowConfig)


class AgentWorkflow(Generic[TConfig], ABC):
    def __init__(self, config: TConfig):
        self.config = config

    @abstractmethod
    async def run(self, ctx: AgentWorkflowContext[TConfig]) -> Any:
        raise NotImplementedError