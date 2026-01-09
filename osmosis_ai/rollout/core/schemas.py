"""Pydantic schemas for Osmosis remote rollout protocol.

This module defines the data models used for communication between
RolloutServer and TrainGate. These schemas are the "single source of truth"
and should be imported by both sides.

Schema Categories:
    - Type Aliases: MessageDict, SamplingParamsDict
    - Tool Schemas: OpenAI-compatible function definitions
    - Rollout Messages: Request/Response for rollout lifecycle
    - Completions Messages: LLM chat completions protocol
    - Metrics: Execution metrics and statistics

Example:
    from osmosis_ai.rollout.core.schemas import (
        RolloutRequest,
        RolloutResponse,
        OpenAIFunctionToolSchema,
    )

    request = RolloutRequest(
        rollout_id="r123",
        server_url="http://localhost:8080",
        messages=[{"role": "user", "content": "Hello"}],
        completion_params={"temperature": 0.7},
    )
"""

from __future__ import annotations

import json
import threading
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Type Aliases
# =============================================================================

MessageDict = Dict[str, Any]
"""Type alias for message dicts in protocol transmission.

Supports the full OpenAI message format including tool_call_id for tool responses.

Example:
    {"role": "tool", "content": "345", "tool_call_id": "call_123"}
"""

SamplingParamsDict = Dict[str, Any]
"""Type alias for sampling parameters dict.

Standard keys: temperature, top_p, max_tokens, stop, logprobs.

Example:
    {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "logprobs": True}
"""


# =============================================================================
# Tool Definition Schemas (OpenAI-compatible)
# =============================================================================
# The following schemas define the structure of tool/function definitions
# that are passed to the LLM. They follow the OpenAI function calling format.
#
# These schemas are adapted from verl/tools/schemas.py
# (Copyright 2023-2024 SGLang Team, Copyright 2025 ModelBest Inc.)
# Source: https://github.com/volcengine/verl/blob/main/verl/tools/schemas.py


class OpenAIFunctionPropertySchema(BaseModel):
    """Schema for a single property in function parameters.

    Follows JSON Schema specification for property definitions.

    Attributes:
        type: JSON Schema type (string, number, boolean, etc.).
        description: Optional description of the property.
        enum: Optional list of allowed values.
    """

    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class OpenAIFunctionParametersSchema(BaseModel):
    """Schema for function parameters following JSON Schema spec.

    Attributes:
        type: Always "object" for function parameters.
        properties: Dictionary of parameter definitions.
        required: List of required parameter names.
    """

    type: str
    properties: Dict[str, OpenAIFunctionPropertySchema]
    required: List[str]


class OpenAIFunctionSchema(BaseModel):
    """Schema for a function definition.

    Attributes:
        name: Function name (should be valid identifier).
        description: Human-readable description for the LLM.
        parameters: JSON Schema for function parameters.
        strict: Whether to enforce strict parameter validation.
    """

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(
            type="object", properties={}, required=[]
        )
    )
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """OpenAI-compatible tool schema.

    Example:
        tool = OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="add",
                description="Add two numbers",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "a": OpenAIFunctionPropertySchema(type="number"),
                        "b": OpenAIFunctionPropertySchema(type="number"),
                    },
                    required=["a", "b"],
                ),
            ),
        )

    Attributes:
        type: Tool type (always "function" for now).
        function: The function schema definition.

    Note:
        When serializing this schema for LLM API calls (e.g., OpenAI, Anthropic),
        always use ``model_dump(exclude_none=True)``. LLM APIs reject ``null``
        values for optional fields like ``enum`` - they expect the field to be
        either an array or absent entirely, not ``null``.

        Example::

            # Correct - excludes None values
            tools = [t.model_dump(exclude_none=True) for t in tool_schemas]

            # Wrong - will include {"enum": null} which causes API errors
            tools = [t.model_dump() for t in tool_schemas]
    """

    type: str
    function: OpenAIFunctionSchema


# =============================================================================
# Tool Call Schemas (OpenAI-compatible)
# =============================================================================
# The following schemas are adapted from verl/tools/schemas.py
# (Copyright 2023-2024 SGLang Team, Copyright 2025 ModelBest Inc.)
# They represent the runtime tool call structures used during agent execution.


class OpenAIFunctionParsedSchema(BaseModel):
    """Parsed function call from LLM output (arguments as JSON string).

    This represents the raw output from the LLM before argument parsing.
    The arguments field contains a JSON string that needs to be parsed.

    Attributes:
        name: The function name to call.
        arguments: JSON string of function arguments (needs parsing).

    Example:
        parsed = OpenAIFunctionParsedSchema(
            name="add",
            arguments='{"a": 5, "b": 3}'
        )
    """

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """Parsed function call with arguments as dict.

    This represents a function call after the arguments JSON string
    has been parsed into a Python dict.

    Attributes:
        name: The function name to call.
        arguments: Parsed function arguments as a dictionary.

    Example:
        call = OpenAIFunctionCallSchema(
            name="add",
            arguments={"a": 5, "b": 3}
        )
    """

    name: str
    arguments: Dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> Tuple["OpenAIFunctionCallSchema", bool]:
        """Parse a function call from LLM output.

        Args:
            parsed_schema: The raw parsed schema with JSON string arguments.

        Returns:
            A tuple of (parsed_call, has_decode_error).
            If decoding fails, arguments will be empty dict and has_decode_error=True.
        """
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return (
            OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments),
            has_decode_error,
        )


class OpenAIFunctionToolCall(BaseModel):
    """Complete tool call structure in OpenAI format.

    This represents a full tool call as returned by the LLM, including
    the unique call ID, type, and function details.

    Attributes:
        id: Unique identifier for this tool call (e.g., "call_abc123").
        type: Tool type, always "function" for function calls.
        function: The function call details.

    Example:
        tool_call = OpenAIFunctionToolCall(
            id="call_abc123",
            type="function",
            function=OpenAIFunctionCallSchema(
                name="add",
                arguments={"a": 5, "b": 3}
            )
        )
    """

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


class ToolResponse(BaseModel):
    """Response from a tool execution.

    Supports multimodal responses including text, images, and videos.
    At least one field should be non-empty for a valid response.

    Attributes:
        text: Text response from the tool.
        image: List of images (for multimodal tools).
        video: List of videos (for multimodal tools).

    Example:
        # Text-only response
        response = ToolResponse(text="The result is 42")

        # Multimodal response
        response = ToolResponse(
            text="Here is the generated image",
            image=[image_data]
        )
    """

    text: Optional[str] = None
    image: Optional[List[Any]] = None
    video: Optional[List[Any]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_media_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that image and video fields are lists if provided."""
        if "image" in values and values["image"] is not None:
            if not isinstance(values["image"], list):
                raise ValueError(
                    f"image must be a list, but got {type(values['image'])}. "
                    f"For single images, wrap in a list: [image]. "
                    f"Example: {{'image': [img1]}} or {{'image': [img1, img2, ...]}}."
                )
        if "video" in values and values["video"] is not None:
            if not isinstance(values["video"], list):
                raise ValueError(
                    f"video must be a list, but got {type(values['video'])}. "
                    f"For single videos, wrap in a list: [video]. "
                    f"Example: {{'video': [video1]}} or {{'video': [video1, video2, ...]}}."
                )
        return values

    def is_empty(self) -> bool:
        """Check if the response has no content."""
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        """Check if the response contains only text."""
        return bool(self.text) and not self.image and not self.video


# =============================================================================
# Rollout Status
# =============================================================================


class RolloutStatus(str, Enum):
    """Status of a rollout execution.

    Values:
        COMPLETED: Rollout finished successfully.
        ERROR: Rollout failed with an error.
    """

    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


# =============================================================================
# Rollout Metrics
# =============================================================================


class RolloutMetrics(BaseModel):
    """Metrics from rollout execution.

    Tracks timing, token usage, and call counts for monitoring
    and optimization purposes.

    Attributes:
        total_latency_ms: Total wall-clock time in milliseconds.
        llm_latency_ms: Time spent waiting for LLM responses.
        tool_latency_ms: Time spent executing tools.
        num_llm_calls: Number of LLM generation calls.
        num_tool_calls: Number of tool executions.
        prompt_tokens: Total input tokens to LLM.
        response_tokens: Total output tokens from LLM.
        max_context_tokens: Maximum context size used.
    """

    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    max_context_tokens: int = 0


# =============================================================================
# Metadata Size Configuration
# =============================================================================

# Default metadata size limit (1MB)
DEFAULT_MAX_METADATA_SIZE_BYTES = 1 * 1024 * 1024

# Configurable metadata size limit (thread-safe)
_max_metadata_size_bytes = DEFAULT_MAX_METADATA_SIZE_BYTES
_max_metadata_size_lock = threading.Lock()


def get_max_metadata_size_bytes() -> int:
    """Get the current maximum metadata size limit in bytes.

    This function is thread-safe.

    Returns:
        Maximum allowed metadata size in bytes.
    """
    with _max_metadata_size_lock:
        return _max_metadata_size_bytes


def set_max_metadata_size_bytes(size_bytes: int) -> None:
    """Set the maximum metadata size limit in bytes.

    This function is thread-safe.

    Args:
        size_bytes: Maximum size in bytes. Must be positive.

    Raises:
        ValueError: If size_bytes is not positive.

    Example:
        # Set to 2MB
        set_max_metadata_size_bytes(2 * 1024 * 1024)
    """
    global _max_metadata_size_bytes
    if size_bytes <= 0:
        raise ValueError("max_metadata_size_bytes must be positive")
    with _max_metadata_size_lock:
        _max_metadata_size_bytes = size_bytes


# =============================================================================
# Rollout Request/Response (RolloutServer <- TrainGate)
# =============================================================================


class RolloutRequest(BaseModel):
    """Request sent to POST /v1/rollout/init to start a rollout.

    TrainGate sends this request to RolloutServer. RolloutServer should
    return 202 Accepted with an InitResponse containing tools for this rollout.

    The rollout continues asynchronously: RolloutServer calls back to
    server_url/v1/chat/completions for LLM generation, and POSTs the final
    RolloutResponse to server_url/v1/rollout/completed when finished.

    Attributes:
        rollout_id: Unique rollout identifier (1-256 characters).
        server_url: TrainGate base URL for callbacks.
        messages: Initial conversation messages.
        completion_params: Sampling parameters (temperature, top_p, etc.).
        tool_server_url: Optional URL for external tool server.
        max_turns: Advisory max LLM calls.
        max_tokens_total: Advisory max total tokens.
        metadata: Optional fine-grained control parameters (max 1MB).
        api_key: Optional API key for authenticating RolloutServer -> TrainGate
            callback requests. If provided, RolloutServer will attach it as an
            HTTP Bearer token when calling:
            - POST {server_url}/v1/chat/completions
            - POST {server_url}/v1/rollout/completed
        idempotency_key: Optional key for retry safety.
    """

    rollout_id: str = Field(min_length=1, max_length=256)
    server_url: str
    messages: List[MessageDict]
    completion_params: SamplingParamsDict
    tool_server_url: Optional[str] = None
    max_turns: int = 10
    max_tokens_total: int = 8192
    metadata: Dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[str] = None
    idempotency_key: Optional[str] = None

    @field_validator("rollout_id")
    @classmethod
    def validate_rollout_id_format(cls, v: str) -> str:
        """Validate rollout_id is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("rollout_id cannot be empty or whitespace only")
        return v

    @field_validator("server_url")
    @classmethod
    def validate_server_url_format(cls, v: str) -> str:
        """Validate server_url is a valid URL with http or https scheme."""
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("server_url must use http or https scheme")
        if not parsed.netloc:
            raise ValueError("server_url must have a valid host")
        return v

    @model_validator(mode="after")
    def validate_metadata_size(self) -> "RolloutRequest":
        """Validate metadata size does not exceed limit."""
        if self.metadata:
            try:
                metadata_json = json.dumps(self.metadata)
                max_size = get_max_metadata_size_bytes()
                if len(metadata_json.encode("utf-8")) > max_size:
                    raise ValueError(
                        f"metadata size exceeds maximum allowed size of "
                        f"{max_size // (1024 * 1024)}MB"
                    )
            except (TypeError, ValueError) as e:
                if "exceeds maximum" in str(e):
                    raise
                raise ValueError(f"metadata must be JSON serializable: {e}")
        return self


class InitResponse(BaseModel):
    """Response from RolloutServer POST /v1/rollout/init endpoint (202 Accepted).

    Contains the tools available for this specific rollout.

    Attributes:
        rollout_id: Echoed back for correlation.
        tools: List of tools available for this rollout.
    """

    rollout_id: str
    tools: List[OpenAIFunctionToolSchema] = Field(default_factory=list)


class RolloutResponse(BaseModel):
    """Response from RolloutServer after completing the rollout.

    Posted to TrainGate's /v1/rollout/completed endpoint.

    Attributes:
        rollout_id: Echoed back for correlation.
        status: COMPLETED or ERROR.
        final_messages: Final conversation messages.
        finish_reason: Why the rollout ended.
        error_message: Error message if status=ERROR.
        reward: Optional precomputed trajectory reward score.
        metrics: Optional execution metrics.
        extra_fields: Additional fields for extensibility.
    """

    rollout_id: str
    status: RolloutStatus
    final_messages: List[MessageDict] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    reward: Optional[float] = None
    metrics: Optional[RolloutMetrics] = None
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Completions Request/Response (RolloutServer -> TrainGate)
# =============================================================================


class CompletionsRequest(BaseModel):
    """OpenAI-compatible completions request with rollout_id extension.

    RolloutServer sends this to TrainGate's /v1/chat/completions endpoint.
    The rollout_id is used to route the request to the correct session.

    Important: Messages should be the FULL conversation history (append-only).

    Attributes:
        model: Model name (ignored, uses loaded model).
        messages: Full conversation message list.
        rollout_id: Custom extension for session routing.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Maximum response tokens.
        stop: Optional stop sequences.
        logprobs: Whether to return log probabilities.
    """

    model: str = "default"
    messages: List[MessageDict]
    rollout_id: str = Field(min_length=1, max_length=256)
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    logprobs: bool = True

    @field_validator("rollout_id")
    @classmethod
    def validate_rollout_id_format(cls, v: str) -> str:
        """Validate rollout_id is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("rollout_id cannot be empty or whitespace only")
        return v


class CompletionsChoice(BaseModel):
    """Single choice in completions response.

    Attributes:
        index: Choice index (usually 0 for single response).
        message: The assistant's message.
        finish_reason: Why generation stopped.
    """

    index: int = 0
    message: MessageDict
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    """Token usage statistics (OpenAI-compatible).

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionsResponse(BaseModel):
    """OpenAI-compatible completions response.

    Attributes:
        id: Request ID.
        object: Object type (always "chat.completion").
        created: Unix timestamp.
        model: Model name.
        choices: List of completion choices.
        usage: Token usage statistics.
        token_ids: Response token IDs (TrainGate internal use only).
        logprobs: Log probabilities (TrainGate internal use only).
        prompt_token_ids: Prompt token IDs (TrainGate internal use only).

    Note:
        The ``token_ids``, ``logprobs``, and ``prompt_token_ids`` fields are
        used internally by TrainGate for training data collection. These fields
        are accumulated in TrainGate's SessionManager and are NOT transmitted
        via HTTP to RolloutServer. RolloutServer implementations should not
        rely on or expect these fields in the response.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str = "default"
    choices: List[CompletionsChoice]
    usage: Optional[CompletionUsage] = None
    # TrainGate internal fields - NOT transmitted via HTTP to RolloutServer.
    # These are accumulated in SessionManager for building AgentLoopOutput.
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    prompt_token_ids: Optional[List[int]] = None
