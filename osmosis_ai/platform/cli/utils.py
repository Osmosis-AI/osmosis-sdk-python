"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.models import (
    RUN_STATUSES_ERROR,
    RUN_STATUSES_IN_PROGRESS,
    RUN_STATUSES_STOPPED,
    RUN_STATUSES_SUCCESS,
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_SUCCESS,
)


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


def format_dataset_status(d: Any, *, for_prompt: bool = False) -> str:
    """Format a dataset status string with optional color styling.

    When *for_prompt* is True, returns a plain text string suitable for
    interactive prompt choices.
    """
    status_info = f"[{d.status}]"
    if d.processing_step:
        status_info = f"[{d.status}: {d.processing_step}]"

    if for_prompt:
        return status_info

    if d.status in STATUSES_SUCCESS:
        color = "green"
    elif d.status in STATUSES_IN_PROGRESS:
        color = "yellow"
    elif d.status in STATUSES_ERROR:
        color = "red"
    else:
        color = None

    if color:
        return console.format_styled(status_info, color)
    return console.escape(status_info)


def format_run_status(r: Any, *, for_prompt: bool = False) -> str:
    """Format a training run status string with optional color styling.

    When *for_prompt* is True, returns a plain text string suitable for
    interactive prompt choices.
    """
    status_info = f"[{r.status}]"
    if r.processing_step:
        pct = (
            f" {r.processing_percent:.0f}%" if r.processing_percent is not None else ""
        )
        status_info = f"[{r.status}: {r.processing_step}{pct}]"

    if for_prompt:
        return status_info

    if r.status in RUN_STATUSES_SUCCESS:
        return console.format_styled(status_info, "green")
    if r.status in RUN_STATUSES_IN_PROGRESS:
        return console.format_styled(status_info, "yellow")
    if r.status in RUN_STATUSES_ERROR:
        return console.format_styled(status_info, "red")
    if r.status in RUN_STATUSES_STOPPED:
        return console.format_styled(status_info, "dim")
    return console.escape(status_info)


def format_processing_step(obj: Any) -> str | None:
    """Format processing step with optional percentage, or None if no step."""
    if not obj.processing_step:
        return None
    pct = (
        f" ({obj.processing_percent:.0f}%)"
        if obj.processing_percent is not None
        else ""
    )
    return f"{obj.processing_step}{pct}"


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
