"""Tests for ExternalLLMClient."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osmosis_ai.rollout.client import CompletionsResult
from osmosis_ai.rollout.core.schemas import OpenAIFunctionToolSchema


class TestExternalLLMClient:
    """Tests for ExternalLLMClient."""

    def test_model_auto_prefix(self) -> None:
        """Test that simple model names get auto-prefixed with openai/."""
        with patch("osmosis_ai.rollout.eval.common.llm_client._get_litellm") as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.eval.common.llm_client import (
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
        with patch("osmosis_ai.rollout.eval.common.llm_client._get_litellm") as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.eval.common.llm_client import (
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
        with patch("osmosis_ai.rollout.eval.common.llm_client._get_litellm") as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.eval.common.llm_client import (
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
        with patch("osmosis_ai.rollout.eval.common.llm_client._get_litellm") as mock:
            mock.return_value = MagicMock()
            from osmosis_ai.rollout.eval.common.llm_client import (
                ExternalLLMClient,
            )

            async with ExternalLLMClient() as client:
                assert isinstance(client, ExternalLLMClient)

    @pytest.mark.asyncio
    async def test_close_runs_litellm_cleanup_once(self) -> None:
        """Test that close() invokes LiteLLM cleanup and is idempotent."""
        mock_litellm = MagicMock()
        mock_litellm.close_litellm_async_clients = AsyncMock()

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient()
            await client.close()
            await client.close()

            mock_litellm.close_litellm_async_clients.assert_awaited_once()

    def test_get_litellm_disables_atexit_cleanup_registration(self) -> None:
        """Test _get_litellm marks cleanup registration as already handled."""
        fake_litellm = types.SimpleNamespace(_async_client_cleanup_registered=False)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from osmosis_ai.rollout.eval.common.llm_client import _get_litellm

            loaded = _get_litellm()

        assert loaded is fake_litellm
        assert fake_litellm._async_client_cleanup_registered is True

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
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient(model="gpt-4o")
            result = await client.chat_completions([{"role": "user", "content": "Hi"}])

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

        received_kwargs: dict[str, Any] = {}

        async def capture_kwargs(**kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            return mock_response

        mock_litellm.acompletion = capture_kwargs

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.llm_client import (
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

    @pytest.mark.asyncio
    async def test_wraps_missing_provider_error_with_compact_hint(self) -> None:
        """Provider inference failures should return concise actionable errors."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))
        mock_litellm.acompletion = AsyncMock(
            side_effect=Exception(
                "litellm.BadRequestError: LLM Provider NOT provided. "
                "Pass in the LLM provider you are trying to call. "
                "You passed model=Qwen/Qwen3-0.6B"
            )
        )

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import (
                ExternalLLMClient,
            )

            client = ExternalLLMClient(
                model="Qwen/Qwen3-0.6B",
                api_base="http://localhost:1234/v1",
            )

            with pytest.raises(SystemicProviderError) as exc_info:
                await client.chat_completions([{"role": "user", "content": "hello"}])

            message = str(exc_info.value)
            assert "Cannot connect to custom endpoint" in message
            assert "base_url='http://localhost:1234/v1'" in message


class TestPreflightCheck:
    """Tests for ExternalLLMClient.preflight_check()."""

    @pytest.mark.asyncio
    async def test_preflight_success(self) -> None:
        """Preflight succeeds when acompletion returns normally."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    model_dump=MagicMock(
                        return_value={"role": "assistant", "content": "h"}
                    )
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            # Should not raise
            await client.preflight_check()

    @pytest.mark.asyncio
    async def test_preflight_auth_failure(self) -> None:
        """Preflight raises SystemicProviderError on auth failure."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        auth_error = mock_litellm.AuthenticationError("Invalid API key")
        mock_litellm.acompletion = AsyncMock(side_effect=auth_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.preflight_check()

    @pytest.mark.asyncio
    async def test_preflight_connection_failure(self) -> None:
        """Preflight raises SystemicProviderError on connection failure."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        conn_error = mock_litellm.APIConnectionError("Connection refused")
        mock_litellm.acompletion = AsyncMock(side_effect=conn_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.preflight_check()

    @pytest.mark.asyncio
    async def test_preflight_rate_limit_passes(self) -> None:
        """Preflight treats RateLimitError as success (endpoint is reachable)."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        rate_error = mock_litellm.RateLimitError("Rate limited")
        mock_litellm.acompletion = AsyncMock(side_effect=rate_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            # Should not raise - rate limit means endpoint is reachable
            await client.preflight_check()

    @pytest.mark.asyncio
    async def test_preflight_404_api_error(self) -> None:
        """Preflight raises SystemicProviderError on 404 API error."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        api_error = mock_litellm.APIError("Model not found")
        api_error.status_code = 404
        mock_litellm.acompletion = AsyncMock(side_effect=api_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.preflight_check()

    @pytest.mark.asyncio
    async def test_preflight_connection_error_wrapped_as_internal_server_error(
        self,
    ) -> None:
        """Preflight catches connection errors wrapped as InternalServerError (APIError subclass)."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        # litellm wraps connection errors as InternalServerError (APIError subclass)
        InternalServerError = type("InternalServerError", (mock_litellm.APIError,), {})
        api_error = InternalServerError(
            "InternalServerError: OpenAIException - Connection error."
        )
        api_error.status_code = 500
        mock_litellm.acompletion = AsyncMock(side_effect=api_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(
                model="gpt-4o", api_base="http://localhost:9999/v1"
            )
            with pytest.raises(SystemicProviderError) as exc_info:
                await client.preflight_check()
            assert "Cannot connect" in str(exc_info.value)


class TestSystemicErrorClassification:
    """Tests for systemic vs per-row error classification in chat_completions."""

    @pytest.mark.asyncio
    async def test_auth_error_is_systemic(self) -> None:
        """AuthenticationError should raise SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        mock_litellm.acompletion = AsyncMock(
            side_effect=mock_litellm.AuthenticationError("Invalid key")
        )

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.chat_completions([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_error_is_systemic(self) -> None:
        """APIConnectionError should raise SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        mock_litellm.acompletion = AsyncMock(
            side_effect=mock_litellm.APIConnectionError("Connection refused")
        )

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.chat_completions([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_rate_limit_is_not_systemic(self) -> None:
        """RateLimitError should raise ProviderError, not SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        mock_litellm.acompletion = AsyncMock(
            side_effect=mock_litellm.RateLimitError("Too many requests")
        )

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import (
                ProviderError,
                SystemicProviderError,
            )
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(ProviderError) as exc_info:
                await client.chat_completions([{"role": "user", "content": "hi"}])

            # Should be ProviderError but NOT SystemicProviderError
            assert not isinstance(exc_info.value, SystemicProviderError)

    @pytest.mark.asyncio
    async def test_api_error_401_is_systemic(self) -> None:
        """APIError with 401 status should raise SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        api_error = mock_litellm.APIError("Unauthorized")
        api_error.status_code = 401
        mock_litellm.acompletion = AsyncMock(side_effect=api_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(SystemicProviderError):
                await client.chat_completions([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_error_wrapped_as_api_error_is_systemic(self) -> None:
        """APIError with 'Connection error' message should raise SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        # Simulate litellm wrapping connection error as InternalServerError
        InternalServerError = type("InternalServerError", (mock_litellm.APIError,), {})
        api_error = InternalServerError(
            "InternalServerError: OpenAIException - Connection error."
        )
        api_error.status_code = 500
        mock_litellm.acompletion = AsyncMock(side_effect=api_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import SystemicProviderError
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(
                model="gpt-4o", api_base="http://localhost:9999/v1"
            )
            with pytest.raises(SystemicProviderError) as exc_info:
                await client.chat_completions([{"role": "user", "content": "hi"}])
            assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_error_500_is_not_systemic(self) -> None:
        """APIError with 500 status should raise ProviderError, not SystemicProviderError."""
        mock_litellm = MagicMock()
        for error_name in (
            "RateLimitError",
            "AuthenticationError",
            "APIError",
            "BudgetExceededError",
            "Timeout",
            "ContextWindowExceededError",
            "APIConnectionError",
        ):
            setattr(mock_litellm, error_name, type(error_name, (Exception,), {}))

        api_error = mock_litellm.APIError("Internal error")
        api_error.status_code = 500
        mock_litellm.acompletion = AsyncMock(side_effect=api_error)

        with patch(
            "osmosis_ai.rollout.eval.common.llm_client._get_litellm",
            return_value=mock_litellm,
        ):
            from osmosis_ai.rollout.eval.common.errors import (
                ProviderError,
                SystemicProviderError,
            )
            from osmosis_ai.rollout.eval.common.llm_client import ExternalLLMClient

            client = ExternalLLMClient(model="gpt-4o")
            with pytest.raises(ProviderError) as exc_info:
                await client.chat_completions([{"role": "user", "content": "hi"}])

            assert not isinstance(exc_info.value, SystemicProviderError)


class TestLiteLLMImportError:
    """Tests for LiteLLM import error handling."""

    def test_missing_litellm_raises_provider_error(self) -> None:
        """Test that missing LiteLLM raises ProviderError."""
        from osmosis_ai.rollout.eval.common.errors import ProviderError

        with (
            patch.dict("sys.modules", {"litellm": None}),
            patch("osmosis_ai.rollout.eval.common.llm_client._get_litellm") as mock,
        ):
            mock.side_effect = ProviderError("LiteLLM is required")

            with pytest.raises(ProviderError) as exc_info:
                from osmosis_ai.rollout.eval.common.llm_client import (
                    ExternalLLMClient,
                )

                ExternalLLMClient()

            assert "LiteLLM" in str(exc_info.value)
