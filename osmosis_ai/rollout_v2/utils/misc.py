from typing import Any


def map_initial_messages_to_content_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
