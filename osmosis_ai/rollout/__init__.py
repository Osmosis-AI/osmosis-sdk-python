"""Osmosis remote rollout SDK.

This module provides a lightweight layer for integrating agent frameworks
with the Osmosis remote rollout protocol. Users implement RolloutAgentLoop
to define their agent logic, and the SDK handles protocol communication.

Features:
    - RolloutAgentLoop base class for implementing agent logic
    - HTTP client for TrainGate communication
    - FastAPI server factory for hosting agents
    - Type-safe configuration with pydantic-settings (optional)

Example:
    from osmosis_ai.rollout import (
        RolloutAgentLoop, RolloutContext, RolloutResult,
        RolloutRequest, OpenAIFunctionToolSchema, create_app,
    )

    class MyAgentLoop(RolloutAgentLoop):
        name = "my_agent"

        def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
            return []  # Define your tools

        async def run(self, ctx: RolloutContext) -> RolloutResult:
            messages = list(ctx.request.messages)
            # Your agent logic here
            return ctx.complete(messages)

    # Create and run server
    app = create_app(MyAgentLoop())
    # uvicorn main:app --port 9000

Optional Features:
    Install optional dependencies for enhanced functionality:

    pip install osmosis-ai[config]  # pydantic-settings configuration
    pip install osmosis-ai[server]  # FastAPI server support
    pip install osmosis-ai[full]    # Everything
"""

# Core classes
from osmosis_ai.rollout.core.base import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
)
from osmosis_ai.rollout.client import (
    CompletionsResult,
    OsmosisLLMClient,
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
from osmosis_ai.rollout.registry import (
    AgentLoopRegistry,
    get_agent_loop,
    list_agent_loops,
    register_agent_loop,
    unregister_agent_loop,
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
from osmosis_ai.rollout.server.app import create_app
from osmosis_ai.rollout.server.serve import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ServeError,
    serve_agent_loop,
    validate_and_report,
)
from osmosis_ai.rollout.validator import (
    AgentLoopValidationError,
    ValidationError,
    ValidationResult,
    validate_agent_loop,
)
from osmosis_ai.rollout.tools import (
    create_tool_error_result,
    create_tool_result,
    execute_tool_calls,
    get_tool_call_info,
    parse_tool_arguments,
    serialize_tool_result,
)
from osmosis_ai.rollout.utils import (
    count_messages_by_role,
    get_message_content,
    get_message_role,
    is_assistant_message,
    is_tool_message,
    is_user_message,
    normalize_stop,
    parse_tool_calls,
)
from osmosis_ai.rollout.network import (
    detect_public_ip,
    PublicIPDetectionError,
    validate_ipv4,
    is_valid_hostname_or_ip,
    is_private_ip,
)

# Configuration
from osmosis_ai.rollout.config import (
    RolloutClientSettings,
    RolloutServerSettings,
    RolloutSettings,
    configure,
    get_settings,
    reset_settings,
)

__all__ = [
    # Core classes
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutResult",
    # Client
    "OsmosisLLMClient",
    "CompletionsResult",
    "LLMClientProtocol",
    # Server
    "create_app",
    "serve_agent_loop",
    "validate_and_report",
    "ServeError",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    # Validation
    "validate_agent_loop",
    "ValidationResult",
    "ValidationError",
    "AgentLoopValidationError",
    # Registry
    "AgentLoopRegistry",
    "register_agent_loop",
    "unregister_agent_loop",
    "get_agent_loop",
    "list_agent_loops",
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
    # Schemas - Type Aliases
    "MessageDict",
    "SamplingParamsDict",
    # Schemas - Configuration
    "DEFAULT_MAX_METADATA_SIZE_BYTES",
    "get_max_metadata_size_bytes",
    "set_max_metadata_size_bytes",
    # Exceptions
    "OsmosisRolloutError",
    "OsmosisTransportError",
    "OsmosisServerError",
    "OsmosisValidationError",
    "OsmosisTimeoutError",
    "AgentLoopNotFoundError",
    "ToolExecutionError",
    "ToolArgumentError",
    # Tool utilities
    "create_tool_result",
    "create_tool_error_result",
    "serialize_tool_result",
    "parse_tool_arguments",
    "get_tool_call_info",
    "execute_tool_calls",
    # Message utilities
    "parse_tool_calls",
    "normalize_stop",
    "get_message_content",
    "get_message_role",
    "is_assistant_message",
    "is_tool_message",
    "is_user_message",
    "count_messages_by_role",
    # Network utilities
    "detect_public_ip",
    "PublicIPDetectionError",
    "validate_ipv4",
    "is_valid_hostname_or_ip",
    "is_private_ip",
    # Configuration
    "RolloutSettings",
    "RolloutClientSettings",
    "RolloutServerSettings",
    "get_settings",
    "configure",
    "reset_settings",
]
