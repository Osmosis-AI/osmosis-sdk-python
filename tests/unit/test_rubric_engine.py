"""Tests for osmosis_ai.eval.rubric.engine's public evaluate_rubric."""

import json
from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.eval.rubric.engine import evaluate_rubric
from osmosis_ai.eval.rubric.types import (
    MissingAPIKeyError,
    RubricResult,
)

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


_COMPLETION_PATCH = "osmosis_ai.eval.rubric.engine._litellm_completion"


class TestEvaluateRubric:
    """Tests for the main evaluate_rubric function."""

    @pytest.fixture()
    def mock_rubric_litellm(self):
        """Mock both litellm module and _litellm_completion for rubric eval tests."""
        mock_completion = MagicMock()
        with (
            patch(
                "osmosis_ai.eval.rubric.engine.litellm.supports_response_schema",
                return_value=True,
            ),
            patch(_COMPLETION_PATCH, mock_completion),
        ):
            yield None, mock_completion

    async def test_solution_str_returns_rubric_result(self, mock_rubric_litellm):
        _, mock_completion = mock_rubric_litellm
        mock_completion.return_value = _create_mock_litellm_response(
            0.85, "Good response"
        )

        result = await evaluate_rubric(
            solution_str="The answer is 42",
            rubric="Score accuracy",
            model="openai/gpt-5.4",
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
                    model="openai/gpt-5.4",
                )

    async def test_empty_rubric_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            await evaluate_rubric(
                solution_str="Some text",
                rubric="",
                model="openai/gpt-5.4",
                api_key="test-key",
            )

    async def test_empty_solution_str_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            await evaluate_rubric(
                solution_str="",
                rubric="Score it",
                model="openai/gpt-5.4",
                api_key="test-key",
            )
