from abc import ABC, abstractmethod

from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.types import AgentWorkflowConfig, AgentWorkflowOutput, Messages


class AgentWorkflow[TConfig: AgentWorkflowConfig](ABC):
    def __init__(self, config: TConfig | None = None):
        self.config = config

    @abstractmethod
    async def run(
        self, ctx: AgentWorkflowContext[TConfig]
    ) -> AgentWorkflowOutput | Messages | None:
        """Run the workflow and hand its output back to the backend.

        Return an ``AgentWorkflowOutput`` (or a bare message list, which is
        wrapped as its ``messages``) to make the return value the trajectory
        source. Return ``None`` to fall back to the sample collected on the
        active ``RolloutContext``.
        """
        raise NotImplementedError
