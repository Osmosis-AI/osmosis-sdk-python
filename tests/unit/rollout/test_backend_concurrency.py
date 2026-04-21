from osmosis_ai.rollout.backend.base import ExecutionBackend


def test_base_backend_max_concurrency_default_zero():
    """Base ExecutionBackend returns 0 (no limit) by default."""

    class StubBackend(ExecutionBackend):
        async def execute(self, request, on_workflow_complete, on_grader_complete=None):
            pass

    backend = StubBackend()
    assert backend.max_concurrency == 0


def test_local_backend_max_concurrency():
    """LocalBackend exposes its limiter's max_concurrent value."""
    from osmosis_ai.rollout.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout.backend.local.backend import LocalBackend
    from osmosis_ai.rollout.context import AgentWorkflowContext

    class StubWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext):
            pass

    backend = LocalBackend(workflow=StubWorkflow)
    # Default concurrency is 4 (from LocalBackend.__init__ when no config)
    assert backend.max_concurrency == 4


def test_local_backend_max_concurrency_from_config():
    """LocalBackend respects workflow_config.concurrency.max_concurrent."""
    from osmosis_ai.rollout.agent_workflow import AgentWorkflow
    from osmosis_ai.rollout.backend.local.backend import LocalBackend
    from osmosis_ai.rollout.context import AgentWorkflowContext
    from osmosis_ai.rollout.types import AgentWorkflowConfig, ConcurrencyConfig

    class StubWorkflow(AgentWorkflow):
        async def run(self, ctx: AgentWorkflowContext):
            pass

    config = AgentWorkflowConfig(
        name="test",
        concurrency=ConcurrencyConfig(max_concurrent=8),
    )
    backend = LocalBackend(workflow=StubWorkflow, workflow_config=config)
    assert backend.max_concurrency == 8
