from typing import Any


def map_initial_messages_to_content_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert plain-string-content messages to content-block-content messages.

    ``[{"role": "user", "content": "Hello"}]`` becomes
    ``[{"role": "user", "content": [{"text": "Hello"}]}]``. Messages whose
    ``content`` is already a list are passed through unchanged so this is
    safe to call on mixed or already-converted inputs.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": [{"text": content}]})
        else:
            converted.append(message)
    return converted
