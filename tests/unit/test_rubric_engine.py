"""Tests for osmosis_ai.rollout.eval.rubric.engine -- provider parsing, prompt building, JSON parsing, and evaluation."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.rollout.eval.rubric.engine import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    _build_user_prompt,
    _default_timeout_for_model,
    _parse_provider,
    _sanitize_json,
    _to_litellm_model,
    evaluate_rubric,
    extract_assistant_content,
)
from osmosis_ai.rollout.eval.rubric.types import (
    MissingAPIKeyError,
    RubricResult,
)

# =============================================================================
# _parse_provider Tests
# =============================================================================


class TestParseProvider:
    """Tests for provider extraction from model strings."""

    def test_openai_prefix(self):
        assert _parse_provider("openai/gpt-4o") == "openai"

    def test_anthropic_prefix(self):
        assert _parse_provider("anthropic/claude-3") == "anthropic"

    def test_bare_gpt_infers_openai(self):
        assert _parse_provider("gpt-4o") == "openai"

    def test_bare_claude_infers_anthropic(self):
        assert _parse_provider("claude-3-haiku") == "anthropic"

    def test_bare_gemini_infers_gemini(self):
        assert _parse_provider("gemini-2-flash") == "gemini"

    def test_bare_grok_infers_xai(self):
        assert _parse_provider("grok-4-fast") == "xai"

    def test_unknown_bare_model_returns_none(self):
        assert _parse_provider("some-random-model") is None

    def test_empty_string_returns_none(self):
        assert _parse_provider("") is None


# =============================================================================
# _to_litellm_model Tests
# =============================================================================


class TestToLitellmModel:
    """Tests for model format conversion."""

    def test_prefixed_model_unchanged(self):
        assert _to_litellm_model("openai/gpt-4o") == "openai/gpt-4o"
        assert _to_litellm_model("openai/gpt-5-mini") == "openai/gpt-5-mini"
        assert _to_litellm_model("anthropic/claude-3") == "anthropic/claude-3"
        assert _to_litellm_model("groq/llama-3.1-70b") == "groq/llama-3.1-70b"

    def test_bare_openai_model_gets_prefix(self):
        assert _to_litellm_model("gpt-4o") == "openai/gpt-4o"
        assert _to_litellm_model("o3-mini") == "openai/o3-mini"

    def test_bare_non_openai_model_unchanged(self):
        assert _to_litellm_model("claude-3-haiku") == "claude-3-haiku"


# =============================================================================
# _default_timeout_for_model Tests
# =============================================================================


class TestDefaultTimeoutForModel:
    """Tests for default timeout calculation."""

    def test_xai_grok4_timeout(self):
        assert _default_timeout_for_model("xai", "grok-4") == 60.0

    def test_openai_default_timeout(self):
        assert (
            _default_timeout_for_model("openai", "gpt-4o")
            == DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

    def test_unknown_provider_returns_default(self):
        assert (
            _default_timeout_for_model("unknown", "some-model")
            == DEFAULT_REQUEST_TIMEOUT_SECONDS
        )


# =============================================================================
# _sanitize_json Tests
# =============================================================================


class TestSanitizeJson:
    """Tests for JSON response parsing."""

    def test_valid_json(self):
        score, explanation = _sanitize_json(
            '{"score": 0.85, "explanation": "Good response"}'
        )
        assert score == 0.85
        assert explanation == "Good response"

    def test_json_with_code_fence(self):
        score, explanation = _sanitize_json(
            '```json\n{"score": 0.9, "explanation": "Test"}\n```'
        )
        assert score == 0.9
        assert explanation == "Test"

    def test_json_with_think_blocks(self):
        raw = '<think>Let me evaluate this...</think>\n{"score": 0.7, "explanation": "Reasonable"}'
        score, explanation = _sanitize_json(raw)
        assert score == 0.7
        assert explanation == "Reasonable"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _sanitize_json("not valid json")

    def test_missing_score_raises(self):
        with pytest.raises(ValueError, match="numeric 'score'"):
            _sanitize_json('{"explanation": "No score"}')


# =============================================================================
# _build_user_prompt Tests
# =============================================================================


class TestBuildUserPrompt:
    """Tests for user prompt construction."""

    def test_basic_case(self):
        prompt = _build_user_prompt(
            rubric_prompt="Score factual accuracy.",
            score_min=0.0,
            score_max=1.0,
            candidate_output="The capital of France is Paris.",
            original_input=None,
            ground_truth=None,
            metadata=None,
        )

        assert "Rubric:" in prompt
        assert "Score range: 0.0 to 1.0." in prompt
        assert "<<<BEGIN_CANDIDATE_OUTPUT>>>" in prompt
        assert "The capital of France is Paris." in prompt
        # No optional sections when not provided
        assert "<<<BEGIN_ORIGINAL_INPUT>>>" not in prompt
        assert "<<<BEGIN_GROUND_TRUTH>>>" not in prompt
        assert "<<<BEGIN_METADATA>>>" not in prompt

    def test_with_ground_truth_and_original_input(self):
        prompt = _build_user_prompt(
            rubric_prompt="Score quality.",
            score_min=0.0,
            score_max=5.0,
            candidate_output="Thank you for your patience!",
            original_input="Please draft a friendly reply.",
            ground_truth="Thanks for waiting!",
            metadata=None,
        )

        assert "<<<BEGIN_ORIGINAL_INPUT>>>" in prompt
        assert "Please draft a friendly reply." in prompt
        assert "<<<BEGIN_GROUND_TRUTH>>>" in prompt
        assert "Thanks for waiting!" in prompt

    def test_with_metadata(self):
        prompt = _build_user_prompt(
            rubric_prompt="Score the tone.",
            score_min=0.0,
            score_max=1.0,
            candidate_output="Hello there.",
            original_input=None,
            ground_truth=None,
            metadata={"notes": "Consider politeness."},
        )

        assert "<<<BEGIN_METADATA>>>" in prompt
        assert "Consider politeness." in prompt


# =============================================================================
# extract_assistant_content Tests
# =============================================================================


class TestExtractAssistantContent:
    """Tests for extracting the last assistant message content."""

    def test_normal_conversation(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        assert extract_assistant_content(messages) == "The answer is 4."

    def test_no_assistant_message(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        assert extract_assistant_content(messages) == ""

    def test_multiple_assistant_messages_returns_last(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
        assert extract_assistant_content(messages) == "Second answer"


# =============================================================================
# evaluate_rubric Tests
# =============================================================================


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


_COMPLETION_PATCH = "osmosis_ai.rollout.eval.rubric.engine._litellm_completion"


class TestEvaluateRubric:
    """Tests for the main evaluate_rubric function."""

    @pytest.fixture()
    def mock_rubric_litellm(self):
        """Mock both litellm module and _litellm_completion for rubric eval tests."""
        mock_litellm = MagicMock()
        mock_completion = MagicMock()
        with (
            patch.dict(sys.modules, {"litellm": mock_litellm}),
            patch(_COMPLETION_PATCH, mock_completion),
        ):
            yield mock_litellm, mock_completion

    async def test_solution_str_returns_rubric_result(self, mock_rubric_litellm):
        _, mock_completion = mock_rubric_litellm
        mock_completion.return_value = _create_mock_litellm_response(
            0.85, "Good response"
        )

        result = await evaluate_rubric(
            solution_str="The answer is 42",
            rubric="Score accuracy",
            model="openai/gpt-4o",
            api_key="test-key",
        )

        assert isinstance(result, RubricResult)
        assert result.score == 0.85
        assert result.explanation == "Good response"
        mock_completion.assert_called_once()

    async def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(MissingAPIKeyError):
                await evaluate_rubric(
                    solution_str="Some text",
                    rubric="Score it",
                    model="openai/gpt-4o",
                )

    async def test_empty_rubric_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            await evaluate_rubric(
                solution_str="Some text",
                rubric="",
                model="openai/gpt-4o",
                api_key="test-key",
            )

    async def test_empty_solution_str_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            await evaluate_rubric(
                solution_str="",
                rubric="Score it",
                model="openai/gpt-4o",
                api_key="test-key",
            )
