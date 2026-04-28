"""Public API for the rollout SDK."""

import importlib

from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.backend import ExecutionBackend, LocalBackend
from osmosis_ai.rollout.context import (
    AgentWorkflowContext,
    GraderContext,
    HarborAgentWorkflowContext,
    RolloutContext,
    get_rollout_context,
)
from osmosis_ai.rollout.grader import Grader
from osmosis_ai.rollout.server.auth import ControllerAuth
from osmosis_ai.rollout.types import (
    AgentWorkflowConfig,
    ConcurrencyConfig,
    ExecutionRequest,
    ExecutionResult,
    GraderCompleteRequest,
    GraderConfig,
    GraderStatus,
    MultiTurnMode,
    RolloutCompleteRequest,
    RolloutErrorCategory,
    RolloutInitRequest,
    RolloutInitResponse,
    RolloutSample,
    RolloutStatus,
)

_LAZY_IMPORTS: dict[str, str] = {
    "create_rollout_server": "osmosis_ai.rollout.server.app",
    "OsmosisRolloutModel": "osmosis_ai.rollout.integrations.agents.strands",
    "OsmosisStrandsAgent": "osmosis_ai.rollout.integrations.agents.strands",
}

_EXTRA_HINTS: dict[str, str] = {
    "create_rollout_server": "pip install osmosis-ai[server]",
    "OsmosisRolloutModel": "pip install osmosis-ai[server]",
    "OsmosisStrandsAgent": "pip install osmosis-ai[server]",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        try:
            module = importlib.import_module(_LAZY_IMPORTS[name])
        except ImportError:
            hint = _EXTRA_HINTS.get(name, "")
            raise ImportError(
                f"'{name}' requires additional dependencies: {hint}"
            ) from None
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentWorkflow",
    "AgentWorkflowConfig",
    "AgentWorkflowContext",
    "ConcurrencyConfig",
    "ControllerAuth",
    "ExecutionBackend",
    "ExecutionRequest",
    "ExecutionResult",
    "Grader",
    "GraderCompleteRequest",
    "GraderConfig",
    "GraderContext",
    "GraderStatus",
    "HarborAgentWorkflowContext",
    "LocalBackend",
    "MultiTurnMode",
    "OsmosisRolloutModel",
    "OsmosisStrandsAgent",
    "RolloutCompleteRequest",
    "RolloutContext",
    "RolloutErrorCategory",
    "RolloutInitRequest",
    "RolloutInitResponse",
    "RolloutSample",
    "RolloutStatus",
    "create_rollout_server",
    "get_rollout_context",
]
