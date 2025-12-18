"""
osmosis-ai: A Python library for LLM training workflows.

Features:
- Reward function validation with @osmosis_reward and @osmosis_rubric decorators
- Remote rollout SDK for integrating agent frameworks with Osmosis training
- Type-safe interfaces for LLM-centric workflows
"""

from .rubric_eval import MissingAPIKeyError, evaluate_rubric
from .rubric_types import ModelNotFoundError, ProviderRequestError
from .utils import osmosis_reward, osmosis_rubric

# Remote rollout SDK exports
from .rollout import (
    # Core classes
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
    # Client
    OsmosisLLMClient,
    CompletionsResult,
    # Server
    create_app,
    # Registry
    register_agent_loop,
    get_agent_loop,
    list_agent_loops,
    # Schemas
    RolloutRequest,
    RolloutResponse,
    InitResponse,
    OpenAIFunctionToolSchema,
    RolloutMetrics,
    # Exceptions
    OsmosisRolloutError,
    OsmosisTransportError,
    OsmosisServerError,
    OsmosisValidationError,
    OsmosisTimeoutError,
)

__all__ = [
    # Reward function decorators
    "osmosis_reward",
    "osmosis_rubric",
    "evaluate_rubric",
    "MissingAPIKeyError",
    "ProviderRequestError",
    "ModelNotFoundError",
    # Remote rollout SDK
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutResult",
    "OsmosisLLMClient",
    "CompletionsResult",
    "create_app",
    "register_agent_loop",
    "get_agent_loop",
    "list_agent_loops",
    "RolloutRequest",
    "RolloutResponse",
    "InitResponse",
    "OpenAIFunctionToolSchema",
    "RolloutMetrics",
    "OsmosisRolloutError",
    "OsmosisTransportError",
    "OsmosisServerError",
    "OsmosisValidationError",
    "OsmosisTimeoutError",
]
