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
from osmosis_ai.rollout.core.llm_client import LLMClientProtocol
from osmosis_ai.rollout.core.schemas import (
    DEFAULT_MAX_METADATA_SIZE_BYTES,
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    CompletionUsage,
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
    # Configuration
    "DEFAULT_MAX_METADATA_SIZE_BYTES",
    "AgentLoopNotFoundError",
    "CompletionUsage",
    "CompletionsChoice",
    # Schemas - Completions
    "CompletionsRequest",
    "CompletionsResponse",
    "InitResponse",
    # Protocol
    "LLMClientProtocol",
    # Type aliases
    "MessageDict",
    "OpenAIFunctionCallSchema",
    "OpenAIFunctionParametersSchema",
    # Schemas - Tool Call (adapted from verl)
    "OpenAIFunctionParsedSchema",
    "OpenAIFunctionPropertySchema",
    "OpenAIFunctionSchema",
    "OpenAIFunctionToolCall",
    # Schemas - Tool Definition
    "OpenAIFunctionToolSchema",
    # Exceptions
    "OsmosisRolloutError",
    "OsmosisServerError",
    "OsmosisTimeoutError",
    "OsmosisTransportError",
    "OsmosisValidationError",
    # Base classes
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutMetrics",
    # Schemas - Request/Response
    "RolloutRequest",
    "RolloutResponse",
    "RolloutResult",
    "RolloutStatus",
    "SamplingParamsDict",
    "ToolArgumentError",
    "ToolExecutionError",
    "ToolResponse",
    "get_max_metadata_size_bytes",
    "set_max_metadata_size_bytes",
]
