from __future__ import annotations

from statistics import mean, pvariance, pstdev
from typing import Any, Collection, Optional, Set

from .errors import CLIError


def coerce_optional_float(value: Any, field_name: str, source_label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise CLIError(
        f"Expected '{field_name}' in {source_label} to be numeric, got {type(value).__name__}."
    )


def collapse_preview_text(value: Any, *, max_length: int = 140) -> Optional[str]:
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.strip().split())
    if not collapsed:
        return None
    if len(collapsed) > max_length:
        collapsed = collapsed[: max_length - 3].rstrip() + "..."
    return collapsed


def calculate_statistics(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {
            "average": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    average = mean(scores)
    variance = pvariance(scores)
    std_dev = pstdev(scores)
    return {
        "average": average,
        "variance": variance,
        "stdev": std_dev,
        "min": min(scores),
        "max": max(scores),
    }


def calculate_stat_deltas(baseline: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, current_value in current.items():
        if key not in baseline:
            continue
        try:
            baseline_value = float(baseline[key])
            current_numeric = float(current_value)
        except (TypeError, ValueError):
            continue
        delta[key] = current_numeric - baseline_value
    return delta


def gather_text_fragments(
    node: Any,
    fragments: list[str],
    *,
    allow_free_strings: bool = False,
    seen: Optional[Set[int]] = None,
    string_key_allowlist: Optional[Collection[str]] = None,
) -> None:
    """Collect textual snippets from nested message-like structures.

    The traversal favours common chat-completions shapes (e.g. ``{"type": "text"}``
    blocks) and avoids indiscriminately pulling in metadata values such as IDs.
    ``allow_free_strings`` controls whether bare strings encountered at the current
    level should be considered textual content (useful for raw message content but
    typically disabled for metadata fields).
    """

    if seen is None:
        seen = set()

    if isinstance(node, str):
        if allow_free_strings:
            stripped = node.strip()
            if stripped:
                fragments.append(stripped)
        return

    if isinstance(node, list):
        for item in node:
            gather_text_fragments(
                item,
                fragments,
                allow_free_strings=allow_free_strings,
                seen=seen,
                string_key_allowlist=string_key_allowlist,
            )
        return

    if not isinstance(node, dict):
        return

    node_id = id(node)
    if node_id in seen:
        return
    seen.add(node_id)

    allowlist = {"text", "value", "message"}
    if string_key_allowlist is not None:
        allowlist = {key.lower() for key in string_key_allowlist}
    else:
        allowlist = {key.lower() for key in allowlist}

    prioritized_keys = ("text", "value")
    handled_keys: Set[str] = {
        "text",
        "value",
        "content",
        "message",
        "parts",
        "input_text",
        "output_text",
        "type",
        "role",
        "name",
        "id",
        "index",
        "finish_reason",
        "reason",
        "tool_call_id",
        "metadata",
    }

    for key in prioritized_keys:
        if key not in node:
            continue
        before_count = len(fragments)
        gather_text_fragments(
            node[key],
            fragments,
            allow_free_strings=True,
            seen=seen,
            string_key_allowlist=string_key_allowlist,
        )
        if len(fragments) > before_count:
            break

    if node.get("type") == "tool_result" and "content" in node:
        gather_text_fragments(
            node["content"],
            fragments,
            allow_free_strings=True,
            seen=seen,
            string_key_allowlist=string_key_allowlist,
        )
    elif "content" in node:
        gather_text_fragments(
            node["content"],
            fragments,
            allow_free_strings=True,
            seen=seen,
            string_key_allowlist=string_key_allowlist,
        )

    for key in ("message", "parts", "input_text", "output_text"):
        if key in node:
            gather_text_fragments(
                node[key],
                fragments,
                allow_free_strings=True,
                seen=seen,
                string_key_allowlist=string_key_allowlist,
            )

    for key, value in node.items():
        if key in handled_keys:
            continue
        if isinstance(value, (list, dict)):
            gather_text_fragments(
                value,
                fragments,
                allow_free_strings=False,
                seen=seen,
                string_key_allowlist=string_key_allowlist,
            )
        elif isinstance(value, str) and key.lower() in allowlist:
            stripped = value.strip()
            if stripped:
                fragments.append(stripped)


def collect_text_fragments(
    node: Any,
    *,
    allow_free_strings: bool = False,
    string_key_allowlist: Optional[Collection[str]] = None,
) -> list[str]:
    fragments: list[str] = []
    gather_text_fragments(
        node,
        fragments,
        allow_free_strings=allow_free_strings,
        seen=set(),
        string_key_allowlist=string_key_allowlist,
    )
    return fragments
