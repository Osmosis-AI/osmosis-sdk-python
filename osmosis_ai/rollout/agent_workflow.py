from abc import ABC, abstractmethod
from typing import Any

from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.types import AgentWorkflowConfig


class AgentWorkflow[TConfig: AgentWorkflowConfig](ABC):
    def __init__(self, config: TConfig | None = None):
        self.config = config

    @abstractmethod
    async def run(self, ctx: AgentWorkflowContext[TConfig]) -> Any:
        raise NotImplementedError
