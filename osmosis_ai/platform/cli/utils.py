"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
from osmosis_ai.platform.auth import AuthenticationExpiredError, load_credentials
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.cli.constants import DEFAULT_LIST_LIMIT

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def require_credentials() -> Credentials:
    """Load valid credentials, raising if not available or expired."""
    from osmosis_ai.platform.cli.constants import MSG_NOT_LOGGED_IN

    credentials = load_credentials()
    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)
    if credentials.is_expired():
        raise AuthenticationExpiredError("Session has expired.")
    return credentials


def resolve_id_prefix(
    prefix: str,
    items: list[Any],
    *,
    entity_name: str = "item",
) -> str:
    """Resolve a short ID prefix to a full ID by matching against a list.

    If *prefix* is already 32+ characters it is returned as-is (assumed to be
    a full UUID).  Otherwise the *items* (each must have an ``.id`` attribute)
    are searched for a unique prefix match.
    """
    if len(prefix) >= 32:
        return prefix

    matches = [item for item in items if item.id.startswith(prefix)]
    if len(matches) == 0:
        raise CLIError(f"No {entity_name} found matching ID prefix '{prefix}'.")
    if len(matches) > 1:
        n = len(matches)
        lines = [f"Ambiguous ID prefix '{prefix}' — matches {n} {entity_name}s:"]
        for m in matches[:5]:
            name = getattr(m, "name", None) or getattr(m, "file_name", None)
            entry = f"  {m.id}"
            if name:
                entry += f"  ({name})"
            lines.append(entry)
        if n > 5:
            lines.append(f"  … and {n - 5} more")
        lines.append("Please provide a longer prefix or the full ID.")
        raise CLIError("\n".join(lines))
    return matches[0].id


def _resolve_entity_id(
    id: str,
    project: str | None,
    workspace_name: str,
    credentials: Credentials,
    *,
    list_fn: Any,
    items_attr: str,
    entity_name: str,
) -> str:
    """Resolve a short ID prefix to a full entity ID via API listing."""
    if len(id) >= 32:
        return id
    from .project import _resolve_project_id

    project_id = _resolve_project_id(project, workspace_name=workspace_name)
    all_items, _ = fetch_all_pages(
        lambda limit, offset: list_fn(
            project_id, limit=limit, offset=offset, credentials=credentials
        ),
        items_attr=items_attr,
    )
    return resolve_id_prefix(id, all_items, entity_name=entity_name)


def resolve_dataset_id(
    id: str,
    credentials: Credentials,
    *,
    client: Any = None,
) -> str:
    """Resolve a short ID prefix to a full dataset ID."""
    if len(id) >= 32:
        return id
    if client is None:
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
    all_datasets, _ = fetch_all_pages(
        lambda limit, offset: client.list_datasets(
            limit=limit, offset=offset, credentials=credentials
        ),
        items_attr="datasets",
    )
    return resolve_id_prefix(id, all_datasets, entity_name="dataset")


def resolve_run_id(
    id: str,
    project: str | None,
    workspace_name: str,
    credentials: Credentials,
    *,
    client: Any = None,
) -> str:
    """Resolve a short ID prefix to a full training run ID."""
    if client is None:
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
    return _resolve_entity_id(
        id,
        project,
        workspace_name,
        credentials,
        list_fn=client.list_training_runs,
        items_attr="training_runs",
        entity_name="training run",
    )


def format_dataset_status(d: Any, *, for_prompt: bool = False) -> str:
    """Format a dataset status string with optional color styling.

    When *for_prompt* is True, returns a plain text string suitable for
    interactive prompt choices.
    """
    status_info = f"[{d.status}]"
    if for_prompt:
        return status_info
    color = entity_status_style(d.status)
    if color:
        return console.format_styled(status_info, color)
    return console.escape(status_info)


def entity_status_style(status: str) -> str | None:
    """Return the Rich style name for a dataset/model status, or None if unstyled."""
    if status in STATUSES_SUCCESS:
        return "green"
    if status in STATUSES_IN_PROGRESS:
        return "yellow"
    if status in STATUSES_ERROR:
        return "red"
    return None


def run_status_style(status: str) -> str | None:
    """Return the Rich style name for a training run status, or None if unstyled."""
    if status in RUN_STATUSES_SUCCESS:
        return "green"
    if status in RUN_STATUSES_IN_PROGRESS:
        return "yellow"
    if status in RUN_STATUSES_ERROR:
        return "red"
    if status in RUN_STATUSES_STOPPED:
        return "dim"
    return None


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

    style = run_status_style(r.status)
    if style:
        return console.format_styled(status_info, style)
    return console.escape(status_info)


def format_date(iso_str: str | None) -> str:
    """Extract YYYY-MM-DD from an ISO 8601 string, or return '' if empty."""
    if not iso_str:
        return ""
    return iso_str[:10]


def format_dim_date(iso_str: str | None) -> str:
    """Format a date as dim-styled text for list displays, or '' if empty."""
    if not iso_str:
        return ""
    return console.format_styled(format_date(iso_str), "dim")


def platform_entity_url(
    ws_name: str, project_name: str | None = None, *segments: str
) -> str:
    """Build a platform URL for a workspace or workspace/project entity."""
    base = f"{PLATFORM_URL}/{ws_name}"
    if project_name:
        base += f"/{project_name}"
    if segments:
        base += "/" + "/".join(segments)
    return base


def build_dataset_detail_rows(ds: Any) -> list[tuple[str, str]]:
    """Build common detail rows for a dataset."""
    rows: list[tuple[str, str]] = [
        ("File", console.escape(ds.file_name)),
        ("ID", ds.id),
        ("Size", format_size(ds.file_size)),
        ("Status", ds.status),
    ]
    step = format_processing_step(ds)
    if step:
        rows.append(("Step", step))
    if ds.error:
        rows.append(("Error", console.escape(ds.error)))
    return rows


def build_run_detail_rows(r: Any) -> list[tuple[str, str]]:
    """Build common detail rows for a training run."""
    rows: list[tuple[str, str]] = [
        ("Name", console.escape(r.name) if r.name else "(unnamed)"),
        ("ID", r.id),
        ("Status", r.status),
    ]
    step = format_processing_step(r)
    if step:
        rows.append(("Step", step))
    rows.append(("Model", console.escape(r.model_name) if r.model_name else "—"))
    if r.eval_accuracy is not None:
        rows.append(("Accuracy", f"{r.eval_accuracy:.4f}"))
    if r.reward_increase_delta is not None:
        rows.append(("Reward Delta", f"{r.reward_increase_delta:+.4f}"))
    if r.error_message:
        rows.append(("Error", console.escape(r.error_message)))
    if r.creator_name:
        rows.append(("Creator", console.escape(r.creator_name)))
    if r.created_at:
        rows.append(("Created", format_date(r.created_at)))
    return rows


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


def fetch_all_pages(
    fetch_fn: Callable[[int, int], Any],
    *,
    items_attr: str,
    page_size: int = 50,
) -> tuple[list[Any], int]:
    """Iterate through API pages until ``has_more`` is False.

    *fetch_fn(limit, offset)* must return an object with a ``has_more`` bool,
    a ``total_count`` int, and an attribute named *items_attr* (a list).

    Returns ``(all_items, total_count)``.
    """
    all_items: list[Any] = []
    offset = 0
    total_count = 0
    while True:
        result = fetch_fn(page_size, offset)
        items = getattr(result, items_attr)
        all_items.extend(items)
        total_count = result.total_count
        if not result.has_more or not items:
            break
        offset += len(items)
    return all_items, total_count


def paginated_fetch(
    fetch_fn: Callable[[int, int], Any],
    *,
    items_attr: str,
    limit: int,
    fetch_all: bool,
) -> tuple[list[Any], int, bool]:
    """Fetch items from a paginated API, respecting ``--all`` / ``--limit``.

    When *fetch_all* is True, exhaustively paginates via :func:`fetch_all_pages`.
    Otherwise, issues a single request with the given *limit*.

    Returns ``(items, total_count, has_more)``.
    """
    if fetch_all:
        items, total_count = fetch_all_pages(fetch_fn, items_attr=items_attr)
        return items, total_count, False
    result = fetch_fn(limit, 0)
    return getattr(result, items_attr), result.total_count, result.has_more


def print_pagination_footer(shown: int, total: int, entity_name: str) -> None:
    """Print a dim hint when a list command truncated its output."""
    if shown >= total:
        return
    console.print(
        f"\nShowing {shown} of {total} {entity_name}."
        " Use --all to show all, or --limit to adjust.",
        style="dim",
    )


def validate_list_options(
    *,
    limit: int,
    all_: bool,
) -> tuple[int, bool]:
    """Validate mutual exclusion between ``--all`` and ``--limit``.

    Returns ``(limit, fetch_all)`` on success, or raises :class:`CLIError`.
    """
    if all_ and limit != DEFAULT_LIST_LIMIT:
        raise CLIError("--all and --limit are mutually exclusive.")
    return limit, all_
