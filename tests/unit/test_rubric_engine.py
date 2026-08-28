"""Tests for osmosis_ai.eval.rubric.engine's public evaluate_rubric."""

from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.eval.rubric.engine import evaluate_rubric
from osmosis_ai.eval.rubric.types import (
    MissingAPIKeyError,
    ProviderRequestError,
    RubricResult,
)


def _response(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_response


@pytest.fixture()
def mock_completion():
    completion = MagicMock()
    with (
        patch(
            "osmosis_ai.eval.rubric.engine.litellm.supports_response_schema",
            return_value=True,
        ),
        patch("osmosis_ai.eval.rubric.engine._litellm_completion", completion),
    ):
        yield completion


@pytest.mark.parametrize(("timeout", "expected_timeout"), [(None, 30.0), (12.5, 12.5)])
async def test_request_contract(
    mock_completion: MagicMock, timeout: float | None, expected_timeout: float
) -> None:
    mock_completion.return_value = _response(
        '```json\n{"score": 0.85, "explanation": "Good response"}\n```'
    )

    result = await evaluate_rubric(
        solution_str="The answer is 42",
        rubric="Score accuracy",
        model="gpt-5.4",
        ground_truth="42",
        original_input="What is six times seven?",
        metadata={"source": "unit"},
        api_key="test-key",
        timeout=timeout,
    )

    assert isinstance(result, RubricResult)
    assert (result.score, result.explanation) == (0.85, "Good response")
    request = mock_completion.call_args.kwargs
    assert request["model"] == "openai/gpt-5.4"
    assert request["timeout"] == expected_timeout
    assert request["api_key"] == "test-key"
    user_prompt = request["messages"][1]["content"]
    for expected in (
        "<<<BEGIN_CANDIDATE_OUTPUT>>>",
        "<<<BEGIN_GROUND_TRUTH>>>",
        "<<<BEGIN_ORIGINAL_INPUT>>>",
        "<<<BEGIN_METADATA>>>",
        "The answer is 42",
        "What is six times seven?",
        '"source": "unit"',
    ):
        assert expected in user_prompt


async def test_reasoning_response_is_parsed_and_bounded(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _response(
        '<think>reasoning</think>\n{"score": 2, "explanation": "Strong"}'
    )

    result = await evaluate_rubric(
        solution_str="Answer",
        rubric="Score it",
        model="openai/gpt-5.4",
        api_key="test-key",
    )

    assert (result.score, result.explanation) == (1.0, "Strong")


@pytest.mark.parametrize(
    ("content", "expected_error"),
    [
        ("not valid json", "not valid JSON"),
        (
            '{"score": "high", "explanation": "No numeric score"}',
            "numeric 'score'",
        ),
        (
            '{"score": 0.5, "explanation": ""}',
            "non-empty 'explanation'",
        ),
    ],
)
async def test_invalid_response_is_provider_error(
    mock_completion: MagicMock, content: str, expected_error: str
) -> None:
    mock_completion.return_value = _response(content)

    with pytest.raises(ProviderRequestError, match=expected_error):
        await evaluate_rubric(
            solution_str="Answer",
            rubric="Score it",
            model="openai/gpt-5.4",
            api_key="test-key",
        )


async def test_missing_api_key_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(MissingAPIKeyError):
            await evaluate_rubric(
                solution_str="Some text",
                rubric="Score it",
                model="openai/gpt-5.4",
            )


@pytest.mark.parametrize(
    ("solution_str", "rubric"), [("Some text", ""), ("", "Score it")]
)
async def test_empty_inputs_raise_type_error(solution_str: str, rubric: str) -> None:
    with pytest.raises(TypeError, match="non-empty string"):
        await evaluate_rubric(
            solution_str=solution_str,
            rubric=rubric,
            model="openai/gpt-5.4",
            api_key="test-key",
        )
