"""Core components for Osmosis rollout SDK.

This module contains the fundamental building blocks:
- Base classes for agent loop implementations
- Pydantic schemas for protocol messages
- Exception hierarchy
- Type definitions

Example:
    from osmosis_ai.rollout.core import (
        RolloutAgentLoop,
        RolloutContext,
        RolloutResult,
        RolloutRequest,
        OsmosisRolloutError,
    )
"""

from osmosis_ai.rollout.core.base import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
)
from osmosis_ai.rollout.core.llm_client import LLMClientProtocol
from osmosis_ai.rollout.core.exceptions import (
    AgentLoopNotFoundError,
    OsmosisRolloutError,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
    ToolArgumentError,
    ToolExecutionError,
)
from osmosis_ai.rollout.core.schemas import (
    CompletionUsage,
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    DEFAULT_MAX_METADATA_SIZE_BYTES,
    InitResponse,
    MessageDict,
    OpenAIFunctionCallSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    SamplingParamsDict,
    ToolResponse,
    get_max_metadata_size_bytes,
    set_max_metadata_size_bytes,
)

__all__ = [
    # Base classes
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutResult",
    # Protocol
    "LLMClientProtocol",
    # Exceptions
    "OsmosisRolloutError",
    "OsmosisTransportError",
    "OsmosisServerError",
    "OsmosisValidationError",
    "OsmosisTimeoutError",
    "AgentLoopNotFoundError",
    "ToolExecutionError",
    "ToolArgumentError",
    # Schemas - Request/Response
    "RolloutRequest",
    "RolloutResponse",
    "InitResponse",
    "RolloutStatus",
    "RolloutMetrics",
    # Schemas - Completions
    "CompletionsRequest",
    "CompletionsResponse",
    "CompletionsChoice",
    "CompletionUsage",
    # Schemas - Tool Definition
    "OpenAIFunctionToolSchema",
    "OpenAIFunctionSchema",
    "OpenAIFunctionParametersSchema",
    "OpenAIFunctionPropertySchema",
    # Schemas - Tool Call (adapted from verl)
    "OpenAIFunctionParsedSchema",
    "OpenAIFunctionCallSchema",
    "OpenAIFunctionToolCall",
    "ToolResponse",
    # Type aliases
    "MessageDict",
    "SamplingParamsDict",
    # Configuration
    "DEFAULT_MAX_METADATA_SIZE_BYTES",
    "get_max_metadata_size_bytes",
    "set_max_metadata_size_bytes",
]
