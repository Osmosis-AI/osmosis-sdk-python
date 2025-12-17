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
from typing import Any, Dict, List, Optional
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
# Tool Schemas (OpenAI-compatible)
# =============================================================================


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
    """

    type: str
    function: OpenAIFunctionSchema


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
        api_key: Optional API key for authenticating callbacks.
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
        metrics: Optional execution metrics.
        extra_fields: Additional fields for extensibility.
    """

    rollout_id: str
    status: RolloutStatus
    final_messages: List[MessageDict] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
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
        token_ids: Response token IDs (for training).
        logprobs: Log probabilities (for training).
        prompt_token_ids: Prompt token IDs (for training).
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str = "default"
    choices: List[CompletionsChoice]
    usage: Optional[CompletionUsage] = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    prompt_token_ids: Optional[List[int]] = None
