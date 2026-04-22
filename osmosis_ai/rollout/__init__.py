"""Public API for the rollout SDK."""

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
from osmosis_ai.rollout.server import ControllerAuth, create_rollout_server
from osmosis_ai.rollout_types import (
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

try:
    from osmosis_ai.rollout.integrations.agents.openai_agents import (
        OsmosisOpenAIAgent,  # noqa: F401
    )

    _openai_exports = ["OsmosisOpenAIAgent"]
except ImportError:
    _openai_exports = []

try:
    from osmosis_ai.rollout.integrations.agents.strands import (
        OsmosisRolloutModel,  # noqa: F401
        OsmosisStrandsAgent,  # noqa: F401
    )

    _strands_exports = ["OsmosisRolloutModel", "OsmosisStrandsAgent"]
except ImportError:
    _strands_exports = []

__all__ = [
    "AgentWorkflow",
    "ControllerAuth",
    "AgentWorkflowConfig",
    "AgentWorkflowContext",
    "ConcurrencyConfig",
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
    "RolloutCompleteRequest",
    "RolloutContext",
    "RolloutErrorCategory",
    "RolloutInitRequest",
    "RolloutInitResponse",
    "RolloutSample",
    "RolloutStatus",
    "create_rollout_server",
    "get_rollout_context",
    *_openai_exports,
    *_strands_exports,
]
