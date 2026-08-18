from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend.local.backend import LocalBackend
from osmosis_ai.rollout.context import AgentWorkflowContext
from osmosis_ai.rollout.types import AgentWorkflowConfig, ConcurrencyConfig


class StubWorkflow(AgentWorkflow):
    async def run(self, ctx: AgentWorkflowContext):
        pass


def test_local_backend_defaults_to_four_in_flight():
    """No config means a 4-slot limiter, which is what /health reports."""
    backend = LocalBackend(workflow=StubWorkflow)
    assert backend.limiter.max_concurrent == 4


def test_local_backend_limiter_follows_config():
    """workflow_config.concurrency.max_concurrent sizes the limiter."""
    config = AgentWorkflowConfig(
        name="test",
        concurrency=ConcurrencyConfig(max_concurrent=8),
    )
    backend = LocalBackend(workflow=StubWorkflow, workflow_config=config)
    assert backend.limiter.max_concurrent == 8
