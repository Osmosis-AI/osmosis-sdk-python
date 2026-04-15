from typing import Any


def map_initial_messages_to_content_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert a list of messages in the form of [{"role": "user", "content": "Hello, world!"}]
    to a list of content blocks in the form of [{"role": "user", "content": [{"text": "Hello, world!"}]}].
    """
    return [
        {
            "role": message["role"],
            "content": [
                {
                    "text": message["content"],
                }
            ],
        }
        for message in messages
    ]
