"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

from typing import Any

from osmosis_ai.cli.errors import CLIError


def resolve_id_prefix(
    prefix: str,
    items: list[Any],
    *,
    entity_name: str = "item",
    has_more: bool = False,
) -> str:
    """Resolve a short ID prefix to a full ID by matching against a list.

    If *prefix* is already 32+ characters it is returned as-is (assumed to be
    a full UUID).  Otherwise the *items* (each must have an ``.id`` attribute)
    are searched for a unique prefix match.

    When *has_more* is True and no match is found, the error message hints that
    more items exist beyond the fetched page.
    """
    if len(prefix) >= 32:
        return prefix

    matches = [item for item in items if item.id.startswith(prefix)]
    if len(matches) == 0:
        hint = f"No {entity_name} found matching ID prefix '{prefix}'."
        if has_more:
            hint += " The full list was too large to search — try a longer prefix or the full ID."
        raise CLIError(hint)
    if len(matches) > 1:
        ids = ", ".join(m.id[:12] for m in matches[:5])
        raise CLIError(
            f"Ambiguous ID prefix '{prefix}' — matches {len(matches)} {entity_name}s: {ids}\n"
            "Please provide a longer prefix."
        )
    return matches[0].id


def format_size(size_bytes: int | float) -> str:
    """Format file size as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return (
                f"{size_bytes:.1f} {unit}"
                if unit != "B"
                else f"{int(size_bytes)} {unit}"
            )
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
