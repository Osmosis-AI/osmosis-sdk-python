"""Osmosis remote rollout SDK.

This module provides a lightweight layer for integrating agent frameworks
with the Osmosis remote rollout protocol. Users implement RolloutAgentLoop
to define their agent logic, and the SDK handles protocol communication.

Features:
    - RolloutAgentLoop base class for implementing agent logic
    - HTTP client for TrainGate communication
    - FastAPI server factory for hosting agents
    - Structured logging with structlog (optional)
    - Distributed tracing with OpenTelemetry (optional)
    - Prometheus metrics collection (optional)
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

    pip install osmosis-ai[config]        # pydantic-settings configuration
    pip install osmosis-ai[logging]       # structlog structured logging
    pip install osmosis-ai[tracing]       # OpenTelemetry tracing
    pip install osmosis-ai[metrics]       # Prometheus metrics
    pip install osmosis-ai[observability] # All observability features
    pip install osmosis-ai[full]          # Everything
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

# Configuration
from osmosis_ai.rollout.config import (
    LoggingSettings,
    MetricsSettings,
    RolloutClientSettings,
    RolloutServerSettings,
    RolloutSettings,
    TracingSettings,
    configure,
    get_settings,
    reset_settings,
)

# Observability
from osmosis_ai.rollout.observability import (
    SpanNames,
    clear_context,
    configure_logging,
    configure_metrics,
    configure_tracing,
    get_logger,
    get_metrics,
    get_rollout_id,
    get_tracer,
    reset_metrics,
    reset_tracing,
    set_rollout_id,
    span,
    trace_async,
)

__all__ = [
    # Core classes
    "RolloutAgentLoop",
    "RolloutContext",
    "RolloutResult",
    # Client
    "OsmosisLLMClient",
    "CompletionsResult",
    # Server
    "create_app",
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
    # Configuration
    "RolloutSettings",
    "RolloutClientSettings",
    "RolloutServerSettings",
    "LoggingSettings",
    "TracingSettings",
    "MetricsSettings",
    "get_settings",
    "configure",
    "reset_settings",
    # Observability - Logging
    "get_logger",
    "configure_logging",
    "get_rollout_id",
    "set_rollout_id",
    "clear_context",
    # Observability - Tracing
    "get_tracer",
    "configure_tracing",
    "reset_tracing",
    "span",
    "trace_async",
    "SpanNames",
    # Observability - Metrics
    "get_metrics",
    "configure_metrics",
    "reset_metrics",
]
