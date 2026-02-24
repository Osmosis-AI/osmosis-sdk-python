"""
osmosis-ai: A Python library for LLM training workflows.

Features:
- Reward function validation with @osmosis_reward and @osmosis_rubric decorators
- Remote rollout SDK for integrating agent frameworks with Osmosis training
- Type-safe interfaces for LLM-centric workflows
"""

from .consts import PACKAGE_VERSION as __version__

# Remote rollout SDK exports
from .rollout import (
    CompletionsResult,
    InitResponse,
    OpenAIFunctionToolSchema,
    # Client
    OsmosisLLMClient,
    # Exceptions
    OsmosisRolloutError,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
    # Core classes
    RolloutAgentLoop,
    RolloutContext,
    RolloutMetrics,
    # Schemas
    RolloutRequest,
    RolloutResponse,
    RolloutResult,
    # Server
    create_app,
    get_agent_loop,
    list_agent_loops,
    # Registry
    register_agent_loop,
)
from .rubric import (
    MissingAPIKeyError,
    ModelNotFoundError,
    ProviderRequestError,
    evaluate_rubric,
)
from .utils import osmosis_reward, osmosis_rubric

__all__ = [
    "CompletionsResult",
    "InitResponse",
    "MissingAPIKeyError",
    "ModelNotFoundError",
    "OpenAIFunctionToolSchema",
    "OsmosisLLMClient",
    "OsmosisRolloutError",
    "OsmosisServerError",
    "OsmosisTimeoutError",
    "OsmosisTransportError",
    "OsmosisValidationError",
    "ProviderRequestError",
    # Remote rollout SDK
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutResult",
    # Version
    "__version__",
    "create_app",
    "evaluate_rubric",
    "get_agent_loop",
    "list_agent_loops",
    # Reward function decorators
    "osmosis_reward",
    "osmosis_rubric",
    "register_agent_loop",
]
