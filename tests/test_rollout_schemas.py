# Copyright 2025 Osmosis AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for osmosis_ai.rollout.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout import (
    CompletionsRequest,
    CompletionUsage,
    DEFAULT_MAX_METADATA_SIZE_BYTES,
    InitResponse,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    OpenAIFunctionToolSchema,
    get_max_metadata_size_bytes,
    set_max_metadata_size_bytes,
)


# =============================================================================
# RolloutRequest Tests
# =============================================================================


def test_rollout_request_valid() -> None:
    """Verify valid RolloutRequest creation."""
    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[{"role": "user", "content": "Hello"}],
        completion_params={"temperature": 0.7},
    )
    assert request.rollout_id == "test-123"
    assert request.server_url == "http://localhost:8080"
    assert len(request.messages) == 1
    assert request.max_turns == 10  # default
    assert request.max_tokens_total == 8192  # default


def test_rollout_request_empty_rollout_id_rejected() -> None:
    """Verify empty rollout_id is rejected."""
    with pytest.raises(ValidationError, match="rollout_id"):
        RolloutRequest(
            rollout_id="",
            server_url="http://localhost:8080",
            messages=[],
            completion_params={},
        )


def test_rollout_request_whitespace_rollout_id_rejected() -> None:
    """Verify whitespace-only rollout_id is rejected."""
    with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
        RolloutRequest(
            rollout_id="   ",
            server_url="http://localhost:8080",
            messages=[],
            completion_params={},
        )


def test_rollout_request_rollout_id_max_length() -> None:
    """Verify rollout_id has max length of 256."""
    # 256 chars should work
    request = RolloutRequest(
        rollout_id="a" * 256,
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )
    assert len(request.rollout_id) == 256

    # 257 chars should fail
    with pytest.raises(ValidationError, match="string_too_long"):
        RolloutRequest(
            rollout_id="a" * 257,
            server_url="http://localhost:8080",
            messages=[],
            completion_params={},
        )


def test_rollout_request_metadata_size_limit() -> None:
    """Verify metadata size limit of 1MB."""
    # Large metadata exceeding 1MB should fail
    large_metadata = {"data": "x" * (1024 * 1024 + 1)}
    with pytest.raises(ValidationError, match="exceeds maximum"):
        RolloutRequest(
            rollout_id="test-123",
            server_url="http://localhost:8080",
            messages=[],
            completion_params={},
            metadata=large_metadata,
        )


def test_max_metadata_size_configurable() -> None:
    """Verify metadata size limit can be configured."""
    original_size = get_max_metadata_size_bytes()
    try:
        # Set to 2MB
        set_max_metadata_size_bytes(2 * 1024 * 1024)
        assert get_max_metadata_size_bytes() == 2 * 1024 * 1024

        # Now 1.5MB metadata should be accepted
        large_metadata = {"data": "x" * (1024 * 1024 + 500000)}
        request = RolloutRequest(
            rollout_id="test-123",
            server_url="http://localhost:8080",
            messages=[],
            completion_params={},
            metadata=large_metadata,
        )
        assert request.metadata == large_metadata
    finally:
        # Restore original size
        set_max_metadata_size_bytes(original_size)


def test_max_metadata_size_default_value() -> None:
    """Verify default metadata size is 1MB."""
    assert DEFAULT_MAX_METADATA_SIZE_BYTES == 1 * 1024 * 1024


def test_set_max_metadata_size_rejects_non_positive() -> None:
    """Verify set_max_metadata_size_bytes rejects non-positive values."""
    with pytest.raises(ValueError, match="must be positive"):
        set_max_metadata_size_bytes(0)

    with pytest.raises(ValueError, match="must be positive"):
        set_max_metadata_size_bytes(-1)


def test_rollout_request_metadata_valid() -> None:
    """Verify valid metadata is accepted."""
    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
        metadata={"key": "value", "nested": {"a": 1, "b": 2}},
    )
    assert request.metadata["key"] == "value"


def test_rollout_request_default_values() -> None:
    """Verify default values are set correctly."""
    request = RolloutRequest(
        rollout_id="test-123",
        server_url="http://localhost:8080",
        messages=[],
        completion_params={},
    )
    assert request.max_turns == 10
    assert request.max_tokens_total == 8192
    assert request.metadata == {}
    assert request.api_key is None
    assert request.tool_server_url is None


# =============================================================================
# InitResponse Tests
# =============================================================================


def test_init_response_with_tools(sample_tool_schema: OpenAIFunctionToolSchema) -> None:
    """Verify InitResponse with tools."""
    response = InitResponse(
        rollout_id="test-123",
        tools=[sample_tool_schema],
    )
    assert response.rollout_id == "test-123"
    assert len(response.tools) == 1
    assert response.tools[0].function.name == "add"


def test_init_response_empty_tools() -> None:
    """Verify InitResponse with no tools."""
    response = InitResponse(rollout_id="test-123", tools=[])
    assert response.rollout_id == "test-123"
    assert response.tools == []


def test_init_response_default_tools() -> None:
    """Verify InitResponse defaults to empty tools."""
    response = InitResponse(rollout_id="test-123")
    assert response.tools == []


# =============================================================================
# RolloutResponse Tests
# =============================================================================


def test_rollout_response_completed() -> None:
    """Verify COMPLETED status RolloutResponse."""
    response = RolloutResponse(
        rollout_id="test-123",
        status=RolloutStatus.COMPLETED,
        final_messages=[{"role": "assistant", "content": "Done"}],
        finish_reason="stop",
    )
    assert response.status == RolloutStatus.COMPLETED
    assert response.finish_reason == "stop"
    assert response.error_message is None


def test_rollout_response_error() -> None:
    """Verify ERROR status RolloutResponse."""
    response = RolloutResponse(
        rollout_id="test-123",
        status=RolloutStatus.ERROR,
        final_messages=[],
        finish_reason="error",
        error_message="Something went wrong",
    )
    assert response.status == RolloutStatus.ERROR
    assert response.error_message == "Something went wrong"


def test_rollout_response_with_metrics() -> None:
    """Verify RolloutResponse with metrics."""
    metrics = RolloutMetrics(
        total_latency_ms=1000.0,
        llm_latency_ms=800.0,
        num_llm_calls=5,
    )
    response = RolloutResponse(
        rollout_id="test-123",
        status=RolloutStatus.COMPLETED,
        metrics=metrics,
    )
    assert response.metrics is not None
    assert response.metrics.total_latency_ms == 1000.0
    assert response.metrics.num_llm_calls == 5


# =============================================================================
# OpenAIFunctionToolSchema Tests
# =============================================================================


def test_tool_schema_function_type() -> None:
    """Verify OpenAIFunctionToolSchema requires type and description."""
    tool = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(name="test_func", description="Test function"),
    )
    assert tool.type == "function"
    assert tool.function.description == "Test function"


def test_tool_schema_with_parameters() -> None:
    """Verify OpenAIFunctionToolSchema with full parameters."""
    tool = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="calculate",
            description="Perform calculation",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "operation": OpenAIFunctionPropertySchema(
                        type="string",
                        enum=["add", "subtract"],
                    ),
                    "a": OpenAIFunctionPropertySchema(type="number"),
                    "b": OpenAIFunctionPropertySchema(type="number"),
                },
                required=["operation", "a", "b"],
            ),
        ),
    )
    assert tool.function.name == "calculate"
    assert "operation" in tool.function.parameters.properties
    assert tool.function.parameters.properties["operation"].enum == ["add", "subtract"]


# =============================================================================
# CompletionsRequest Tests
# =============================================================================


def test_completions_request_valid() -> None:
    """Verify valid CompletionsRequest creation."""
    request = CompletionsRequest(
        rollout_id="test-123",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert request.rollout_id == "test-123"
    assert request.model == "default"
    assert request.temperature == 1.0
    assert request.logprobs is True


def test_completions_request_default_values() -> None:
    """Verify CompletionsRequest default values."""
    request = CompletionsRequest(
        rollout_id="test-123",
        messages=[],
    )
    assert request.temperature == 1.0
    assert request.top_p == 1.0
    assert request.max_tokens == 512
    assert request.stop is None
    assert request.logprobs is True


def test_completions_request_rollout_id_validation() -> None:
    """Verify CompletionsRequest validates rollout_id."""
    with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
        CompletionsRequest(
            rollout_id="   ",
            messages=[],
        )


def test_completions_request_custom_params() -> None:
    """Verify CompletionsRequest with custom params."""
    request = CompletionsRequest(
        rollout_id="test-123",
        messages=[],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1024,
        stop=["STOP"],
        logprobs=False,
    )
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.max_tokens == 1024
    assert request.stop == ["STOP"]
    assert request.logprobs is False


# =============================================================================
# RolloutMetrics Tests
# =============================================================================


def test_rollout_metrics_default_values() -> None:
    """Verify RolloutMetrics default values are zero."""
    metrics = RolloutMetrics()
    assert metrics.total_latency_ms == 0.0
    assert metrics.llm_latency_ms == 0.0
    assert metrics.tool_latency_ms == 0.0
    assert metrics.num_llm_calls == 0
    assert metrics.num_tool_calls == 0
    assert metrics.prompt_tokens == 0
    assert metrics.response_tokens == 0
    assert metrics.max_context_tokens == 0


def test_rollout_metrics_all_fields() -> None:
    """Verify RolloutMetrics with all fields set."""
    metrics = RolloutMetrics(
        total_latency_ms=5000.0,
        llm_latency_ms=4000.0,
        tool_latency_ms=500.0,
        num_llm_calls=10,
        num_tool_calls=5,
        prompt_tokens=1000,
        response_tokens=500,
        max_context_tokens=4096,
    )
    assert metrics.total_latency_ms == 5000.0
    assert metrics.llm_latency_ms == 4000.0
    assert metrics.tool_latency_ms == 500.0
    assert metrics.num_llm_calls == 10
    assert metrics.num_tool_calls == 5
    assert metrics.prompt_tokens == 1000
    assert metrics.response_tokens == 500
    assert metrics.max_context_tokens == 4096


# =============================================================================
# CompletionUsage Tests
# =============================================================================


def test_completion_usage_defaults() -> None:
    """Verify CompletionUsage default values."""
    usage = CompletionUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_completion_usage_all_fields() -> None:
    """Verify CompletionUsage with all fields set."""
    usage = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150


# =============================================================================
# RolloutStatus Tests
# =============================================================================


def test_rollout_status_values() -> None:
    """Verify RolloutStatus enum values."""
    assert RolloutStatus.COMPLETED.value == "COMPLETED"
    assert RolloutStatus.ERROR.value == "ERROR"


def test_rollout_status_is_string_enum() -> None:
    """Verify RolloutStatus can be compared with strings."""
    # RolloutStatus inherits from str, so can compare directly
    assert RolloutStatus.COMPLETED == "COMPLETED"
    assert RolloutStatus.ERROR == "ERROR"
