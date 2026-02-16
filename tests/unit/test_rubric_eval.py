from osmosis_ai.rubric_eval import _build_user_prompt, _select_text


def test_build_user_prompt_basic_blocks() -> None:
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


def test_build_user_prompt_with_optional_sections() -> None:
    prompt = _build_user_prompt(
        rubric_prompt="Score the tone.",
        score_min=0.0,
        score_max=5.0,
        candidate_output="Thank you for your patience!",
        original_input="Please draft a friendly reply.",
        ground_truth="Thanks for waiting!",
        metadata={"notes": "Consider politeness."},
    )

    assert "<<<BEGIN_ORIGINAL_INPUT>>>" in prompt
    assert "<<<BEGIN_GROUND_TRUTH>>>" in prompt
    assert "<<<BEGIN_METADATA>>>" in prompt
    assert "Consider politeness." in prompt


def test_select_text_prefers_first_non_empty() -> None:
    assert _select_text(None, "", "  value ", "fallback") == "value"
    assert _select_text(None, "  ") is None
