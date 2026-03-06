from abc import ABC, abstractmethod
from typing import Any, List, Dict
from osmosis_ai.rollout_v2.types import AgentWorkflowConfig
from osmosis_ai.rollout_v2.context import AgentWorkflowContext

class AgentWorkflow[TConfig: AgentWorkflowConfig](ABC):
    def __init__(self, config: TConfig):
        self.config = config
        
    @abstractmethod
    async def run(self, ctx: AgentWorkflowContext[TConfig]) -> Any:
        raise NotImplementedError