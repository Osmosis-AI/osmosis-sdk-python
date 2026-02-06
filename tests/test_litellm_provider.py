"""Tests for LiteLLM-based rubric evaluation."""

import json
import sys
from unittest.mock import MagicMock, patch

import litellm
import pytest

from osmosis_ai.rubric_eval import (
    _to_litellm_model,
    _default_timeout_for_model,
    _sanitize_json,
    evaluate_rubric,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from osmosis_ai.rubric_types import ModelNotFoundError, ProviderRequestError


class TestToLitellmModel:
    """Tests for model format conversion."""

    def test_openai_gpt5_uses_responses_prefix(self):
        assert _to_litellm_model("openai", "gpt-5-mini") == "openai/responses/gpt-5-mini"
        assert _to_litellm_model("openai", "gpt-5") == "openai/responses/gpt-5"
        assert _to_litellm_model("OpenAI", "GPT-5-turbo") == "openai/responses/GPT-5-turbo"

    def test_openai_other_models_standard_prefix(self):
        assert _to_litellm_model("openai", "gpt-4o") == "openai/gpt-4o"
        assert _to_litellm_model("openai", "gpt-4-turbo") == "openai/gpt-4-turbo"
        assert _to_litellm_model("openai", "o1-preview") == "openai/o1-preview"

    def test_anthropic_models(self):
        assert _to_litellm_model("anthropic", "claude-sonnet-4-5-20250929") == "anthropic/claude-sonnet-4-5-20250929"
        assert _to_litellm_model("anthropic", "claude-3-opus-20240229") == "anthropic/claude-3-opus-20240229"

    def test_xai_models(self):
        assert _to_litellm_model("xai", "grok-4-fast") == "xai/grok-4-fast"
        assert _to_litellm_model("xai", "grok-2") == "xai/grok-2"

    def test_gemini_models(self):
        assert _to_litellm_model("gemini", "gemini-2.0-flash") == "gemini/gemini-2.0-flash"
        assert _to_litellm_model("gemini", "gemini-1.5-pro") == "gemini/gemini-1.5-pro"

    def test_other_providers(self):
        assert _to_litellm_model("cerebras", "llama-3.1-70b") == "cerebras/llama-3.1-70b"
        assert _to_litellm_model("openrouter", "meta-llama/llama-3-70b") == "openrouter/meta-llama/llama-3-70b"

    def test_empty_provider_returns_model_only(self):
        assert _to_litellm_model("", "gpt-4o") == "gpt-4o"

    def test_case_insensitive(self):
        assert _to_litellm_model("OPENAI", "gpt-5-mini") == "openai/responses/gpt-5-mini"
        assert _to_litellm_model("Anthropic", "claude-3-opus") == "anthropic/claude-3-opus"


class TestDefaultTimeoutForModel:
    """Tests for default timeout calculation."""

    def test_xai_grok4_timeout(self):
        assert _default_timeout_for_model("xai", "grok-4-fast") == 60.0
        assert _default_timeout_for_model("xai", "grok-4") == 60.0

    def test_xai_other_timeout(self):
        assert _default_timeout_for_model("xai", "grok-2") == 45.0

    def test_openai_gpt5_timeout(self):
        assert _default_timeout_for_model("openai", "gpt-5-mini") == 45.0
        assert _default_timeout_for_model("openai", "gpt-5") == 45.0

    def test_openai_other_timeout(self):
        assert _default_timeout_for_model("openai", "gpt-4o") == DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_gemini_timeout(self):
        assert _default_timeout_for_model("gemini", "gemini-2.0-flash") == 45.0

    def test_cerebras_timeout(self):
        assert _default_timeout_for_model("cerebras", "llama-4-scout-17b") == 60.0

    def test_openrouter_timeout(self):
        assert _default_timeout_for_model("openrouter", "meta-llama/llama-4-maverick") == 60.0

    def test_anthropic_timeout(self):
        assert _default_timeout_for_model("anthropic", "claude-3-opus") == DEFAULT_REQUEST_TIMEOUT_SECONDS


class TestSanitizeJson:
    """Tests for JSON response parsing."""

    def test_valid_json(self):
        score, explanation = _sanitize_json('{"score": 0.85, "explanation": "Good response"}')
        assert score == 0.85
        assert explanation == "Good response"

    def test_json_with_code_fence(self):
        score, explanation = _sanitize_json('```json\n{"score": 0.9, "explanation": "Test"}\n```')
        assert score == 0.9
        assert explanation == "Test"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _sanitize_json("not valid json")

    def test_missing_score_raises(self):
        with pytest.raises(ValueError, match="numeric 'score'"):
            _sanitize_json('{"explanation": "No score"}')

    def test_missing_explanation_raises(self):
        with pytest.raises(ValueError, match="'explanation'"):
            _sanitize_json('{"score": 0.5}')


def _create_mock_litellm_response(score: float, explanation: str) -> MagicMock:
    """Helper to create a mock LiteLLM response."""
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"score": score, "explanation": explanation})
                }
            }
        ]
    }
    return mock_response


class TestEvaluateRubric:
    """Tests for the main evaluate_rubric function."""

    def test_evaluate_rubric_success(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(0.85, "Good response")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score accuracy",
                solution_str="The answer is 42",
                model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
            )

        assert result == 0.85
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0

    def test_evaluate_rubric_gpt5_no_temperature(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(0.9, "Excellent")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            evaluate_rubric(
                rubric="Score quality",
                solution_str="Test response",
                model_info={"provider": "openai", "model": "gpt-5-mini", "api_key": "test-key"},
            )

        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/responses/gpt-5-mini"
        assert "temperature" not in call_kwargs

    def test_evaluate_rubric_anthropic(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(0.9, "Excellent")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score quality",
                solution_str="Well written response",
                model_info={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_key": "test-key"},
            )

        assert result == 0.9
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5-20250929"

    def test_evaluate_rubric_xai(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(0.8, "Good")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score response",
                solution_str="Some response",
                model_info={"provider": "xai", "model": "grok-4-fast", "api_key": "test-key"},
            )

        assert result == 0.8
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "xai/grok-4-fast"

    def test_evaluate_rubric_return_details(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(0.65, "Detailed explanation")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score accuracy",
                solution_str="Some answer",
                model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                return_details=True,
            )

        assert result["score"] == 0.65
        assert result["explanation"] == "Detailed explanation"
        assert "raw" in result

    def test_evaluate_rubric_score_clamping_max(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(1.5, "Over max")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score",
                solution_str="Response",
                model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
            )

        assert result == 1.0  # Clamped to max

    def test_evaluate_rubric_score_clamping_min(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(-0.5, "Under min")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score",
                solution_str="Response",
                model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
            )

        assert result == 0.0  # Clamped to min

    def test_evaluate_rubric_api_error(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = litellm.APIError(
            message="API rate limit exceeded",
            llm_provider="openai",
            model="gpt-4o",
            status_code=429,
        )
        mock_litellm.APIError = litellm.APIError
        mock_litellm.NotFoundError = litellm.NotFoundError

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ProviderRequestError) as exc_info:
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                )

        assert "rate limit" in str(exc_info.value).lower()

    def test_evaluate_rubric_invalid_json_response(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "Not valid JSON"}}]
        }
        mock_litellm.completion.return_value = mock_response

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ProviderRequestError) as exc_info:
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                )

        assert "not valid JSON" in str(exc_info.value)

    def test_evaluate_rubric_empty_response(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        mock_litellm.completion.return_value = mock_response

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ProviderRequestError) as exc_info:
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                )

        assert "did not include any content" in str(exc_info.value)

    def test_evaluate_rubric_model_not_found_404(self):
        """NotFoundError from litellm should raise ModelNotFoundError."""
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = litellm.NotFoundError(
            message="The model `no-such-model` does not exist",
            llm_provider="openai",
            model="no-such-model",
        )
        mock_litellm.APIError = litellm.APIError
        mock_litellm.NotFoundError = litellm.NotFoundError

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ModelNotFoundError) as exc_info:
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "openai", "model": "no-such-model", "api_key": "test-key"},
                )

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "no-such-model"

    def test_evaluate_rubric_model_not_found_is_provider_request_error(self):
        """ModelNotFoundError should be catchable as ProviderRequestError."""
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = litellm.NotFoundError(
            message="Not found",
            llm_provider="anthropic",
            model="bad-model",
        )
        mock_litellm.APIError = litellm.APIError
        mock_litellm.NotFoundError = litellm.NotFoundError

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ProviderRequestError):
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "anthropic", "model": "bad-model", "api_key": "test-key"},
                )

    def test_evaluate_rubric_non_404_error_is_not_model_not_found(self):
        """Non-NotFoundError should raise ProviderRequestError, not ModelNotFoundError."""
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = litellm.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4o",
        )
        mock_litellm.APIError = litellm.APIError
        mock_litellm.NotFoundError = litellm.NotFoundError

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            with pytest.raises(ProviderRequestError) as exc_info:
                evaluate_rubric(
                    rubric="Score",
                    solution_str="Response",
                    model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                )

        assert not isinstance(exc_info.value, ModelNotFoundError)

    def test_evaluate_rubric_custom_score_range(self):
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = _create_mock_litellm_response(7.5, "Good")

        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            result = evaluate_rubric(
                rubric="Score from 0-10",
                solution_str="Response",
                model_info={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                score_min=0.0,
                score_max=10.0,
            )

        assert result == 7.5
