from __future__ import annotations

from osmosis_ai.rubric_eval import _collect_text_from_message


def test_collect_text_skips_metadata_fields() -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Hello world",
                "id": "chunk-1",
                "finish_reason": "stop",
            }
        ],
    }

    assert _collect_text_from_message(message) == "Hello world"


def test_collect_text_includes_nested_tool_results() -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call-123",
                "content": [
                    {
                        "type": "output_text",
                        "text": {"value": "Primary answer."},
                    },
                    {
                        "type": "text",
                        "text": " Additional context. ",
                    },
                ],
            }
        ],
    }

    assert _collect_text_from_message(message) == "Primary answer. Additional context."


def test_collect_text_preserves_order_across_blocks() -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "First sentence."},
            {
                "type": "output_text",
                "text": {
                    "value": "Second sentence.",
                    "content": [{"type": "text", "text": " (nested)"}],
                },
            },
            {
                "type": "message",
                "parts": [
                    {"type": "text", "text": " Third sentence"},
                    {"type": "text", "text": " with wrapper."},
                ],
            },
        ],
    }

    assert (
        _collect_text_from_message(message)
        == "First sentence. Second sentence. (nested) Third sentence with wrapper."
    )


def test_collect_text_ignores_metadata_strings() -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Valid content.",
                "finish_reason": "stop",
                "role": "assistant",
                "name": "response",
                "id": "abc123",
            },
            {
                "type": "tool_result",
                "content": [
                    {
                        "type": "text",
                        "text": "Follow-up content.",
                        "reason": "additional",
                        "tool_call_id": "tool-001",
                    }
                ],
            },
        ],
    }

    assert _collect_text_from_message(message) == "Valid content. Follow-up content."


def test_collect_text_handles_non_string_values_gracefully() -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Lead text",
                "metadata": {"score": 0.75, "tags": ["info", {"nested": "value"}]},
            },
            {
                "type": "message",
                "content": [
                    {"type": "text", "text": " trailing text "},
                    {"type": "text", "text": 1234},  # type: ignore[arg-type]
                ],
            },
        ],
    }

    assert _collect_text_from_message(message) == "Lead text trailing text"
