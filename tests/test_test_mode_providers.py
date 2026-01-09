"""Tests for test_mode providers and registry."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema, RolloutMetrics
from osmosis_ai.rollout.test_mode.providers import (
    get_provider,
    list_providers,
    register_provider,
)
from osmosis_ai.rollout.test_mode.providers.base import TestLLMClient


class TestProviderRegistry:
    """Tests for provider registration and lookup."""

    def test_list_providers_includes_openai(self) -> None:
        """Test that OpenAI provider is registered."""
        providers = list_providers()
        assert "openai" in providers

    def test_get_provider_openai(self) -> None:
        """Test getting OpenAI provider class."""
        provider_class = get_provider("openai")
        assert provider_class.provider_name == "openai"

    def test_get_provider_unknown_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown_provider")
        assert "Unknown provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_register_custom_provider(self) -> None:
        """Test registering a custom provider."""

        class CustomTestClient(TestLLMClient):
            provider_name = "custom_test"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        register_provider(CustomTestClient)

        # Should be available via get_provider
        retrieved = get_provider("custom_test")
        assert retrieved is CustomTestClient

    def test_register_provider_without_name_raises(self) -> None:
        """Test that registering without provider_name raises error."""

        class NoNameClient(TestLLMClient):
            # Missing provider_name

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        with pytest.raises(ValueError) as exc_info:
            register_provider(NoNameClient)
        assert "provider_name" in str(exc_info.value)


class TestTestLLMClient:
    """Tests for TestLLMClient base class."""

    def test_set_and_clear_tools(self) -> None:
        """Test setting and clearing tools."""

        class MockClient(TestLLMClient):
            provider_name = "mock"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        client = MockClient()

        # Initially no tools
        assert client._tools is None

        # Set tools
        tools = [
            OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
        ]
        client.set_tools(tools)
        assert client._tools is not None
        assert len(client._tools) == 1
        assert client._tools[0]["function"]["name"] == "test_tool"

        # Clear tools
        client.clear_tools()
        assert client._tools is None

    def test_metrics_tracking(self) -> None:
        """Test metrics tracking methods."""

        class MockClient(TestLLMClient):
            provider_name = "mock"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        client = MockClient()

        # Initial metrics should be zero
        metrics = client.get_metrics()
        assert metrics.llm_latency_ms == 0.0
        assert metrics.num_llm_calls == 0
        assert metrics.prompt_tokens == 0
        assert metrics.response_tokens == 0

        # Record some usage
        client._record_usage(latency_ms=100.0, prompt_tokens=50, completion_tokens=30)
        client._record_usage(latency_ms=150.0, prompt_tokens=60, completion_tokens=40)

        # Check accumulated metrics
        metrics = client.get_metrics()
        assert metrics.llm_latency_ms == 250.0
        assert metrics.num_llm_calls == 2
        assert metrics.prompt_tokens == 110
        assert metrics.response_tokens == 70

        # Reset metrics
        client.reset_metrics()
        metrics = client.get_metrics()
        assert metrics.llm_latency_ms == 0.0
        assert metrics.num_llm_calls == 0

    @pytest.mark.asyncio
    async def test_tools_auto_injection(self) -> None:
        """Test that tools are auto-injected into chat_completions kwargs."""
        received_kwargs: Dict[str, Any] = {}

        class MockClient(TestLLMClient):
            provider_name = "mock"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                nonlocal received_kwargs
                received_kwargs = kwargs
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        client = MockClient()

        # Set tools
        tools = [
            OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "my_tool",
                        "description": "A tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
        ]
        client.set_tools(tools)

        # Call chat_completions - tools should be auto-injected
        await client.chat_completions([{"role": "user", "content": "hello"}])

        assert "tools" in received_kwargs
        assert received_kwargs["tools"][0]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_explicit_tools_not_overridden(self) -> None:
        """Test that explicitly provided tools are not overridden."""
        received_kwargs: Dict[str, Any] = {}

        class MockClient(TestLLMClient):
            provider_name = "mock"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                nonlocal received_kwargs
                received_kwargs = kwargs
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

        client = MockClient()

        # Set tools via set_tools
        tools = [
            OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "set_tool",
                        "description": "Set tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
        ]
        client.set_tools(tools)

        # Call with explicit tools - should use explicit, not set_tools
        explicit_tools = [{"type": "function", "function": {"name": "explicit_tool"}}]
        await client.chat_completions(
            [{"role": "user", "content": "hello"}], tools=explicit_tools
        )

        assert received_kwargs["tools"][0]["function"]["name"] == "explicit_tool"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager protocol."""
        close_called = False

        class MockClient(TestLLMClient):
            provider_name = "mock"

            async def _do_completion(
                self, messages: List[Dict[str, Any]], **kwargs: Any
            ) -> CompletionsResult:
                return CompletionsResult(
                    message={"role": "assistant", "content": "test"},
                    token_ids=[],
                    logprobs=[],
                    usage={},
                    finish_reason="stop",
                )

            async def close(self) -> None:
                nonlocal close_called
                close_called = True

        async with MockClient() as client:
            assert isinstance(client, MockClient)

        assert close_called


class TestOpenAITestClient:
    """Tests for OpenAI provider."""

    def test_missing_api_key_raises(self) -> None:
        """Test that missing API key raises ProviderError."""
        from osmosis_ai.rollout.test_mode.exceptions import ProviderError

        # Clear env var if set
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderError) as exc_info:
                from osmosis_ai.rollout.test_mode.providers.openai import (
                    OpenAITestClient,
                )

                OpenAITestClient()
            assert "API key required" in str(exc_info.value)

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        from osmosis_ai.rollout.test_mode.providers.openai import OpenAITestClient

        client = OpenAITestClient(api_key="test-key", model="gpt-4o-mini")
        assert client.model == "gpt-4o-mini"

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        from osmosis_ai.rollout.test_mode.providers.openai import OpenAITestClient

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            client = OpenAITestClient()
            assert client.model == "gpt-4o"  # default model

    @pytest.mark.asyncio
    async def test_do_completion_returns_completions_result(self) -> None:
        """Test that _do_completion returns proper CompletionsResult."""
        from osmosis_ai.rollout.test_mode.providers.openai import OpenAITestClient

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    model_dump=MagicMock(
                        return_value={"role": "assistant", "content": "Hello!"}
                    )
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        client = OpenAITestClient(api_key="test-key")
        client._client = AsyncMock()
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client._do_completion([{"role": "user", "content": "Hi"}])

        assert isinstance(result, CompletionsResult)
        assert result.message["role"] == "assistant"
        assert result.message["content"] == "Hello!"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
