from osmosis_ai.eval.controller.messages import preprocess_controller_messages


def test_preprocess_content_none_becomes_empty_string() -> None:
    messages = [{"role": "assistant", "content": None, "ignored": "x"}]

    assert preprocess_controller_messages(messages) == [
        {"role": "assistant", "content": ""}
    ]


def test_preprocess_list_content_uses_first_text_like_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
            "tool_calls": [{"id": "ignored"}],
        }
    ]

    assert preprocess_controller_messages(messages) == [
        {"role": "user", "content": "first"}
    ]


def test_preprocess_list_content_skips_non_text_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "first text"},
            ],
        }
    ]

    assert preprocess_controller_messages(messages) == [
        {"role": "user", "content": "first text"}
    ]


def test_preprocess_list_content_falls_back_to_json_string() -> None:
    messages = [{"role": "user", "content": [{"z": 1, "a": {"url": "x"}}]}]

    assert preprocess_controller_messages(messages) == [
        {"role": "user", "content": '{"a":{"url":"x"},"z":1}'}
    ]


def test_preprocess_missing_role_uses_unknown_for_diagnostics() -> None:
    assert preprocess_controller_messages([{"content": "hello"}]) == [
        {"role": "unknown", "content": "hello"}
    ]


def test_preprocess_preserves_openai_tool_message_linkage() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a":2,"b":3}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "6",
            "tool_call_id": "call_123",
        },
    ]

    assert preprocess_controller_messages(messages) == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a":2,"b":3}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "6",
            "tool_call_id": "call_123",
        },
    ]
