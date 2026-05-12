"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console, console
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
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
    load_credentials,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    load_subscription_status,
    save_subscription_status,
)
from osmosis_ai.platform.cli.project_context import (
    GitProjectContext,
    LocalProjectContext,
    resolve_git_project_context,
    resolve_local_project_context,
)
from osmosis_ai.platform.cli.workspace_context import (
    WorkspaceContext,
    resolve_linked_workspace_context,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


def platform_call[T](
    message: str,
    call: Callable[[], T],
    *,
    output_console: Console | None = None,
) -> T:
    """Run a platform request while showing a consistent CLI loading status."""
    status_console = output_console or console
    with status_console.spinner(message):
        return call()


def require_credentials() -> Credentials:
    """Load valid credentials, raising if not available or expired."""
    from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

    credentials = load_credentials()
    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)
    if credentials.is_expired():
        raise AuthenticationExpiredError()
    return credentials


def require_workspace_context() -> WorkspaceContext:
    """Resolve the current linked Osmosis project workspace."""
    return resolve_linked_workspace_context()


def require_git_project_context() -> GitProjectContext:
    """Resolve the current Git-scoped Osmosis project for platform commands."""
    return resolve_git_project_context()


def require_local_project_context(
    *, require_scaffold: bool = True
) -> LocalProjectContext:
    """Resolve the current Git project for local-only commands."""
    return resolve_local_project_context(require_scaffold=require_scaffold)


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


def platform_entity_url(ws_name: str, *segments: str) -> str:
    """Build a platform URL for a workspace entity."""
    base = f"{PLATFORM_URL}/{ws_name}"
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
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[Any], int]:
    """Iterate through API pages using server-provided ``next_offset``.

    *fetch_fn(limit, offset)* must return an object with a ``next_offset``
    (int | None), a ``total_count`` int, and an attribute named *items_attr*
    (a list).

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
        if result.next_offset is None or not items:
            break
        offset = result.next_offset
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
    if all_ and limit != DEFAULT_PAGE_SIZE:
        raise CLIError("--all and --limit are mutually exclusive.")
    return limit, all_


def _require_auth(
    *,
    workspace_name: str | None = None,
) -> tuple[str, Credentials]:
    """Check that user is authenticated for legacy bootstrap flows."""
    if workspace_name is None:
        raise CLIError(
            "This command requires a linked Osmosis project. "
            "Run 'osmosis project link' from the project root."
        )
    credentials = require_credentials()
    return workspace_name, credentials


def _require_subscription(*, workspace_id: str, workspace_name: str) -> None:
    """Check that a workspace has an active subscription.

    Uses cached status with TTL. If the cache is expired, stale, or False,
    refreshes from the platform to avoid blocking users who just subscribed.
    """
    from osmosis_ai.platform.cli.constants import CACHE_TTL_SECONDS

    cached = load_subscription_status(workspace_name, max_age=CACHE_TTL_SECONDS)
    if cached is True:
        return

    refreshed = False
    api_reached = False
    workspace_found = False
    # Intentionally do NOT suppress AuthenticationExpiredError here: a real
    # 401 must surface as "session expired" rather than be silently swallowed
    # and reported below as the misleading "subscription required" error
    # when a stale `False` subscription cache is present.
    with contextlib.suppress(PlatformAPIError, OSError):
        credentials = require_credentials()
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
        info = platform_call(
            "Checking subscription...",
            lambda: client.refresh_workspace_info(
                credentials=credentials,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
                cleanup_on_401=False,
            ),
        )
        api_reached = True
        workspace_found = info.get("found", False)
        has_subscription = info.get("has_subscription")
        if has_subscription is not None:
            save_subscription_status(workspace_name, bool(has_subscription))
            refreshed = True

    status = load_subscription_status(workspace_name, max_age=CACHE_TTL_SECONDS)
    if status is True:
        return

    if not refreshed and status is None:
        if api_reached and not workspace_found:
            raise CLIError(
                f"Workspace '{workspace_name}' ({workspace_id}) not found. "
                "It may have been deleted, renamed, or become inaccessible.\n"
                "  Run 'osmosis auth login' if you logged in as a different user, "
                "or relink this project:\n"
                "    osmosis project unlink\n"
                "    osmosis project link"
            )
        return

    raise CLIError(
        "Your workspace requires an active subscription for this action.\n"
        f"  Upgrade at: {PLATFORM_URL}/{workspace_name}/settings/billing"
    )
