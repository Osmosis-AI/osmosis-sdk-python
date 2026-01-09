"""Tests for ExternalLLMClient."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema


class TestExternalLLMClient:
    """Tests for ExternalLLMClient."""

    def test_model_auto_prefix(self) -> None:
        """Test that simple model names get auto-prefixed with openai/."""
        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm"
        ) as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            # Simple model name should be prefixed
            client = ExternalLLMClient(model="gpt-4o")
            assert client.model == "openai/gpt-4o"

            # Already prefixed model should not be changed
            client2 = ExternalLLMClient(model="anthropic/claude-sonnet-4-20250514")
            assert client2.model == "anthropic/claude-sonnet-4-20250514"

    def test_set_and_clear_tools(self) -> None:
        """Test setting and clearing tools."""
        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm"
        ) as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient()

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
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
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
        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm"
        ) as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient()

            # Initial metrics should be zero
            metrics = client.get_metrics()
            assert metrics.llm_latency_ms == 0.0
            assert metrics.num_llm_calls == 0
            assert metrics.prompt_tokens == 0
            assert metrics.response_tokens == 0

            # Simulate recording usage
            client._llm_latency_ms = 100.0
            client._num_llm_calls = 1
            client._prompt_tokens = 50
            client._response_tokens = 30

            # Add more
            client._llm_latency_ms += 150.0
            client._num_llm_calls += 1
            client._prompt_tokens += 60
            client._response_tokens += 40

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
    async def test_context_manager(self) -> None:
        """Test async context manager protocol."""
        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm"
        ) as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            async with ExternalLLMClient() as client:
                assert isinstance(client, ExternalLLMClient)

    @pytest.mark.asyncio
    async def test_chat_completions_calls_litellm(self) -> None:
        """Test that chat_completions calls LiteLLM correctly."""
        mock_litellm = MagicMock()
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
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient(model="gpt-4o")
            result = await client.chat_completions(
                [{"role": "user", "content": "Hi"}]
            )

            assert isinstance(result, CompletionsResult)
            assert result.message["role"] == "assistant"
            assert result.message["content"] == "Hello!"
            assert result.finish_reason == "stop"
            assert result.usage["prompt_tokens"] == 10
            assert result.usage["completion_tokens"] == 5

            # Check that metrics were recorded
            metrics = client.get_metrics()
            assert metrics.num_llm_calls == 1
            assert metrics.prompt_tokens == 10
            assert metrics.response_tokens == 5

    @pytest.mark.asyncio
    async def test_tools_auto_injection(self) -> None:
        """Test that tools are auto-injected into chat_completions."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    model_dump=MagicMock(
                        return_value={"role": "assistant", "content": "test"}
                    )
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        received_kwargs: Dict[str, Any] = {}

        async def capture_kwargs(**kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            return mock_response

        mock_litellm.acompletion = capture_kwargs

        with patch(
            "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.test_mode.external_llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient()

            # Set tools
            tools = [
                OpenAIFunctionToolSchema.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": "my_tool",
                            "description": "A tool",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    }
                )
            ]
            client.set_tools(tools)

            # Call chat_completions - tools should be auto-injected
            await client.chat_completions([{"role": "user", "content": "hello"}])

            assert "tools" in received_kwargs
            assert received_kwargs["tools"][0]["function"]["name"] == "my_tool"


class TestLiteLLMImportError:
    """Tests for LiteLLM import error handling."""

    def test_missing_litellm_raises_provider_error(self) -> None:
        """Test that missing LiteLLM raises ProviderError."""
        from osmosis_ai.rollout.test_mode.exceptions import ProviderError

        with patch.dict("sys.modules", {"litellm": None}):
            with patch(
                "osmosis_ai.rollout.test_mode.external_llm_client._get_litellm"
            ) as mock:
                mock.side_effect = ProviderError("LiteLLM is required")

                with pytest.raises(ProviderError) as exc_info:
                    from osmosis_ai.rollout.test_mode.external_llm_client import (
                        ExternalLLMClient,
                    )

                    ExternalLLMClient()

                assert "LiteLLM" in str(exc_info.value)
