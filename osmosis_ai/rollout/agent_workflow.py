from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from osmosis_ai.rollout_v2.context import AgentWorkflowContext
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig

TConfig = TypeVar("TConfig", bound=AgentWorkflowConfig)


class AgentWorkflow(Generic[TConfig], ABC):
    def __init__(self, config: TConfig | None = None):
        self.config = config

    @abstractmethod
    async def run(self, ctx: AgentWorkflowContext[TConfig]) -> Any:
        raise NotImplementedError
