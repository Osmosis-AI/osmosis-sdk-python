"""
Example rubric functions using the @osmosis_rubric decorator.
"""

from osmosis_ai import osmosis_rubric


def _make_message(role: str, text: str, block_type: str) -> dict:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": block_type, "text": text}],
    }


def _analyze_marketing_claim(
    rubric: str,
    messages: list,
    ground_truth: str | None = None,
    system_message: str | None = None,
    extra_info: dict = None,
) -> dict:
    """Helper that surfaces issues and scoring for marketing claim compliance."""
    assistant_turn = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if not assistant_turn:
        return {
            "rubric": rubric,
            "score": 0.0,
            "issues": ["Conversation did not contain an assistant response."],
            "assistant_summary": "",
            "system_message": system_message,
        }

    assistant_text = " ".join(
        block["text"]
        for block in assistant_turn["content"]
        if isinstance(block, dict) and block.get("type") == "output_text"
    )

    issues = []
    score = 1.0

    if ground_truth and ground_truth.lower() not in assistant_text.lower():
        issues.append("Assistant did not cite the approved marketing fact.")
        score -= 0.5

    if extra_info and "forbidden_terms" in extra_info:
        forbidden_terms = [term.lower() for term in extra_info["forbidden_terms"]]
        for term in forbidden_terms:
            if term and term in assistant_text.lower():
                issues.append(f"Assistant used forbidden term '{term}'.")
                score -= 0.25

    return {
        "rubric": rubric,
        "score": max(score, 0.0),
        "issues": issues,
        "assistant_summary": assistant_text,
        "system_message": system_message,
    }


@osmosis_rubric
def marketing_claim_compliance(
    rubric: str,
    messages: list,
    ground_truth: str | None = None,
    system_message: str | None = None,
    extra_info: dict = None,
) -> float:
    """Check that the assistant cites the approved marketing claim and avoids forbidden terms."""
    analysis = _analyze_marketing_claim(
        rubric=rubric,
        messages=messages,
        ground_truth=ground_truth,
        system_message=system_message,
        extra_info=extra_info,
    )
    return analysis["score"]


def _format_report(label: str, analysis: dict) -> str:
    issues = analysis["issues"]
    summary = analysis["assistant_summary"] or "N/A"
    lines = [
        f"{label}",
        f"  Score: {analysis['score']:.2f}",
        f"  Assistant summary: {summary}",
    ]
    if issues:
        lines.append("  Issues:")
        for issue in issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  Issues: none")
    return "\n".join(lines)


if __name__ == "__main__":
    rubric_description = "Assistant must reference the approved product claim and avoid forbidden terms."
    system_instruction = "You are a helpful sales assistant for the Osmosis Pro Router."
    extra = {"forbidden_terms": ["refund", "guarantee"]}

    passing_conversation = [
        _make_message("system", system_instruction, "input_text"),
        _make_message("user", "Why should I upgrade to the Osmosis Pro Router?", "input_text"),
        _make_message(
            "assistant",
            "The Osmosis Pro Router delivers certified gigabit speeds and keeps your network secure with automatic updates.",
            "output_text",
        ),
    ]

    failing_conversation = [
        _make_message("system", system_instruction, "input_text"),
        _make_message("user", "Can I get a refund if I do not like it?", "input_text"),
        _make_message(
            "assistant",
            "We guarantee you will love it, and refunds might be possible.",
            "output_text",
        ),
    ]

    print("Passing conversation result:")
    passing_analysis = _analyze_marketing_claim(
        rubric=rubric_description,
        messages=passing_conversation,
        ground_truth="certified gigabit speeds",
        system_message=system_instruction,
        extra_info=extra,
    )
    print(_format_report("Passing conversation", passing_analysis))

    print("\nFailing conversation result:")
    failing_analysis = _analyze_marketing_claim(
        rubric=rubric_description,
        messages=failing_conversation,
        ground_truth="certified gigabit speeds",
        system_message=system_instruction,
        extra_info=extra,
    )
    print(_format_report("Failing conversation", failing_analysis))
