"""Extract chat messages from the trajectory documents agents leave behind.

Shared by the in-container grader runner and the host-side backend, so it
must not import Harbor.
"""

from __future__ import annotations

from typing import Any

ATIF_ROLE_BY_SOURCE = {"user": "user", "agent": "assistant", "system": "system"}


def messages_from_trajectory(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept ATIF (steps) or raw harness formats (a messages list)."""
    if isinstance(document.get("messages"), list):
        return document["messages"]
    messages = []
    for step in document.get("steps") or []:
        content = step.get("message")
        if content is not None:
            role = ATIF_ROLE_BY_SOURCE.get(step.get("source"), "tool")
            messages.append({"role": role, "content": content})
    return messages
