"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import Console, console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.display import format_local_datetime
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
    load_credentials,
)
from osmosis_ai.platform.cli.workspace_directory_context import (
    GitWorkspaceDirectoryContext,
    resolve_git_workspace_directory_context,
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


def require_git_workspace_directory_context() -> GitWorkspaceDirectoryContext:
    """Resolve the current Git-scoped Osmosis workspace directory for platform commands."""
    return resolve_git_workspace_directory_context()


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


def build_dataset_detail_rows(ds: Any) -> list[tuple[str, str]]:
    """Build common detail rows for a dataset."""
    rows: list[tuple[str, str]] = [
        ("File", console.escape(ds.file_name)),
        ("ID", ds.id),
        ("Status", ds.status.replace("_", " ").title()),
    ]
    file_format = getattr(ds, "file_format", None)
    original_format = getattr(ds, "original_file_format", None)
    if file_format:
        fmt = file_format.upper()
        if original_format and original_format != file_format:
            fmt += f" (uploaded as {original_format.upper()})"
        rows.append(("File Format", fmt))
    if getattr(ds, "row_count", None) is not None:
        rows.append(("Total Rows", f"{ds.row_count:,}"))
    size_str = format_size(ds.file_size)
    original_size = getattr(ds, "original_file_size", None)
    if original_size:
        size_str += f" (uncompressed: {format_size(original_size)})"
    rows.append(("Size", size_str))
    if ds.created_at:
        rows.append(("Uploaded", format_local_datetime(ds.created_at)))
    if getattr(ds, "creator_name", None):
        rows.append(("Uploaded By", console.escape(ds.creator_name)))
    return rows


def build_run_detail_rows(r: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Name", console.escape(r.name) if r.name else "(unnamed)"),
        ("ID", r.id),
        ("Status", r.status.replace("_", " ").title()),
    ]
    if r.created_at:
        rows.append(("Submitted", format_local_datetime(r.created_at)))
    if r.creator_name:
        rows.append(("Submitted By", console.escape(r.creator_name)))
    rows.append(("Dataset", console.escape(r.dataset_name) if r.dataset_name else "—"))
    rows.append(("Base Model", console.escape(r.model_name) if r.model_name else "—"))
    rows.append(("Rollout", console.escape(r.rollout_name) if r.rollout_name else "—"))
    return rows


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


def print_remote_fetch_notice(
    workspace_directory: Path,
    *,
    pinned_commit_sha: str | None,
) -> tuple[list[str], list[str]]:
    """Remind the user that remote submissions pull *code* from the connected
    Git remote while reading *config values* from the local TOML file.

    The platform resolves code from the Platform-connected repository (or
    fetches a pinned commit) before the run starts, so local *code* changes
    that haven't been pushed will silently be ignored. The config TOML
    passed to the submit command, by contrast, is read from disk and its
    values are sent verbatim in the submit payload — local edits to the
    config take effect immediately, even if they are uncommitted.

    Returns ``(notes, warnings)`` as plain-text lists so callers can surface
    the same context in non-rich modes (e.g. the JSON error envelope when
    ``--yes`` is missing). The Rich panel is rendered only when the output
    format is Rich.
    """
    from osmosis_ai.cli.output import OutputFormat, get_output_context
    from osmosis_ai.platform.cli.workspace_repo import summarize_local_git_state

    state = summarize_local_git_state(workspace_directory)

    warnings: list[str] = []
    if state is not None:
        if state.is_dirty:
            warnings.append(
                "Uncommitted changes detected — code edits won't be picked up "
                "(only the config file above is read locally)."
            )
        if state.has_upstream and state.ahead > 0:
            commits_word = "commit" if state.ahead == 1 else "commits"
            warnings.append(
                f"{state.ahead} unpushed {commits_word} ahead of upstream — "
                "push code before submitting."
            )
        elif state.branch is not None and not state.has_upstream:
            warnings.append(
                f"Branch '{state.branch}' has no upstream — "
                "push code and set tracking before submitting."
            )

    notes: list[str] = []
    if pinned_commit_sha:
        notes.append(
            f"Osmosis will fetch commit {pinned_commit_sha} from the "
            "Platform-connected repository."
        )
        notes.append("Make sure that commit is pushed to origin.")
    else:
        notes.append("Osmosis will fetch code from the Platform-connected repository.")
        if state is not None and state.branch and state.head_sha:
            notes.append(f"Local branch: {state.branch} @ {state.head_sha[:8]}")
        notes.append("Make sure your code changes are committed and pushed.")
        warnings.append(
            "Platform source selection may differ from your local branch when no "
            "commit_sha is set."
        )
    notes.append(
        "Config values come from your local TOML file and are submitted "
        "as-is — uncommitted edits to the config still apply."
    )

    if get_output_context().format is OutputFormat.rich:
        body_lines: list[str] = []
        if pinned_commit_sha:
            body_lines.append(
                f"Osmosis will fetch commit [bold]{console.escape(pinned_commit_sha)}[/bold] "
                "from the Platform-connected repository."
            )
            body_lines.append("Make sure that commit is pushed to origin.")
        else:
            body_lines.append(
                "Osmosis will fetch code from the Platform-connected repository."
            )
            if state is not None and state.branch and state.head_sha:
                body_lines.append(
                    f"Local: [bold]{console.escape(state.branch)}[/bold] @ "
                    f"[dim]{console.escape(state.head_sha[:8])}[/dim]"
                )
            body_lines.append("Make sure your code changes are committed and pushed.")

        body_lines.append("")
        body_lines.append(
            "[dim]Config values above come from your local TOML file and are "
            "submitted as-is — uncommitted edits to the config still apply.[/dim]"
        )

        if warnings:
            body_lines.append("")
            for warning in warnings:
                body_lines.append(f"[yellow]• {console.escape(warning)}[/yellow]")

        style = "yellow" if warnings else "blue"
        title = "Push before submitting" if warnings else "Before you submit"
        console.panel(title, "\n".join(body_lines), style=style)

    return notes, warnings
