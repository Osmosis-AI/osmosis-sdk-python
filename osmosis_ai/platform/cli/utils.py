"""Shared utility functions for platform CLI commands."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import DetailSection, ListColumn, ListResult
from osmosis_ai.cli.output.display import format_local_datetime
from osmosis_ai.platform.api.models import (
    DEPLOYMENT_STATUSES_INACTIVE,
    DEPLOYMENT_STATUSES_SUCCESS,
    EVAL_RUN_STATUSES_ERROR,
    EVAL_RUN_STATUSES_IN_PROGRESS,
    EVAL_RUN_STATUSES_PENDING,
    EVAL_RUN_STATUSES_STOPPED,
    EVAL_RUN_STATUSES_SUCCESS,
    RUN_STATUSES_ERROR,
    RUN_STATUSES_IN_PROGRESS,
    RUN_STATUSES_PENDING,
    RUN_STATUSES_STOPPED,
    RUN_STATUSES_SUCCESS,
    STATUSES_ACTIVE,
    STATUSES_ERROR,
    STATUSES_INACTIVE,
    STATUSES_PENDING,
    STATUSES_SUCCESS,
)
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    load_credentials,
)
from osmosis_ai.platform.cli.workspace_directory_context import (
    GitWorkspaceDirectoryContext,
    git_result_context,
    resolve_git_workspace_directory_context,
)
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from osmosis_ai.platform.api.models import LogsPage
    from osmosis_ai.platform.auth.credentials import Credentials


def make_progress(
    completed: int | float | None,
    total: int | float | None,
    unit: str,
) -> dict[str, Any] | None:
    """Build a single source of truth for a run's progress.

    Returns ``None`` unless ``total`` is a positive integer. ``completed`` is
    clamped into ``[0, total]`` so the human-rendered string and the JSON
    ``summary`` are derived from the same numbers and can never disagree or show
    ``N / M`` with ``N > M``. Callers that cannot trust ``total`` as an upper
    bound (e.g. when it is only a lower bound) should widen ``total`` before
    calling.
    """
    if completed is None or total is None:
        return None
    total_int = int(total)
    if total_int <= 0:
        return None
    return {
        "completed": min(max(0, int(completed)), total_int),
        "total": total_int,
        "unit": unit,
    }


def format_progress(progress: dict[str, Any] | None) -> str | None:
    """Render a progress dict produced by :func:`make_progress` as text."""
    if not isinstance(progress, dict):
        return None
    completed = progress.get("completed")
    total = progress.get("total")
    unit = progress.get("unit")
    if (
        not isinstance(completed, int)
        or not isinstance(total, int)
        or not isinstance(unit, str)
    ):
        return None
    return f"{completed:,} / {total:,} {unit}"


def platform_call[T](message: str, call: Callable[[], T]) -> T:
    """Run a platform request while showing a consistent CLI loading status.

    Progress renders on stderr through the active output context, matching the
    other command domains (train/eval/model/...), so stdout stays clean for
    piping and JSON/plain modes stay silent.
    """
    from osmosis_ai.cli.output import get_output_context

    with get_output_context().status(message):
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


# Ordered ``(statuses, style)`` buckets per domain. The first bucket whose set
# contains the status wins, so order matters when a status is a member of more
# than one set. Colors mirror the platform UI status dots: amber = waiting,
# blue = active work, green = success, red = error, dim = stopped/inactive.
_StatusStyleMap = Sequence[tuple[frozenset[str], str]]

_DATASET_STATUS_STYLES: _StatusStyleMap = (
    (STATUSES_SUCCESS, "green"),
    (STATUSES_PENDING, "orange3"),
    (STATUSES_ACTIVE, "blue"),
    (STATUSES_ERROR, "red"),
    (STATUSES_INACTIVE, "dim"),
)
_RUN_STATUS_STYLES: _StatusStyleMap = (
    (RUN_STATUSES_SUCCESS, "green"),
    (RUN_STATUSES_PENDING, "orange3"),
    (RUN_STATUSES_IN_PROGRESS, "blue"),
    (RUN_STATUSES_ERROR, "red"),
    (RUN_STATUSES_STOPPED, "dim"),
)
_DEPLOYMENT_STATUS_STYLES: _StatusStyleMap = (
    (DEPLOYMENT_STATUSES_SUCCESS, "green"),
    (DEPLOYMENT_STATUSES_INACTIVE, "dim"),
)
_EVAL_STATUS_STYLES: _StatusStyleMap = (
    (EVAL_RUN_STATUSES_SUCCESS, "green"),
    (EVAL_RUN_STATUSES_PENDING, "orange3"),
    (EVAL_RUN_STATUSES_IN_PROGRESS, "blue"),
    (EVAL_RUN_STATUSES_ERROR, "red"),
    (EVAL_RUN_STATUSES_STOPPED, "dim"),
)


def format_status_token(
    status: str,
    style_map: _StatusStyleMap,
    *,
    for_prompt: bool = False,
) -> str:
    """Render a ``[status]`` token, Rich-styled by the first matching bucket.

    *style_map* is an ordered sequence of ``(statuses, style)`` pairs; the first
    pair whose set contains *status* wins. List the more specific bucket first
    when a status belongs to several sets (see :data:`_EVAL_STATUS_STYLES`).

    When *for_prompt* is True, returns the plain ``[status]`` label with no Rich
    markup, suitable for interactive prompt choices.
    """
    label = f"[{status}]"
    if for_prompt:
        return label
    for statuses, style in style_map:
        if status in statuses:
            return console.format_styled(label, style)
    return console.escape(label)


def format_dataset_status(d: Any, *, for_prompt: bool = False) -> str:
    """Format a dataset/model status token with optional Rich styling."""
    return format_status_token(d.status, _DATASET_STATUS_STYLES, for_prompt=for_prompt)


def format_run_status(r: Any, *, for_prompt: bool = False) -> str:
    """Format a training run status token with optional Rich styling."""
    return format_status_token(r.status, _RUN_STATUS_STYLES, for_prompt=for_prompt)


def format_deployment_status(status: str | None) -> str:
    """Format a LoRA model deployment status token with Rich styling.

    ``None`` (never deployed) renders as an em dash.
    """
    if status is None:
        return "—"
    return format_status_token(status, _DEPLOYMENT_STATUS_STYLES)


def format_eval_status(run: Any) -> str:
    """Format an evaluation run status token with Rich styling."""
    return format_status_token(run.status, _EVAL_STATUS_STYLES)


def format_reward(reward: float | None) -> str:
    """Format a training reward to two decimals, em dash when unset."""
    if reward is None:
        return "—"
    return f"{reward:.2f}"


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


def build_logs_result(
    *,
    title: str,
    page: LogsPage,
    context: GitWorkspaceDirectoryContext,
    next_step_hint: str,
) -> ListResult:
    """Build the shared ``ListResult`` for a logs page (train/eval/dataset).

    Cursor pagination: the server reports no total, so ``total_count`` is this
    page's size and ``has_more`` means older entries exist beyond ``next_cursor``.
    """
    items = [
        {
            "timestamp": entry.timestamp,
            "level": entry.level,
            "step": entry.step,
            "message": entry.message,
            "details": entry.details,
        }
        for entry in page.logs
    ]
    return ListResult(
        title=title,
        items=items,
        total_count=len(items),
        has_more=page.next_cursor is not None,
        next_offset=None,
        extra={"next_cursor": page.next_cursor, **git_result_context(context)},
        columns=[
            ListColumn(key="timestamp", label="Time", no_wrap=True, ratio=2),
            ListColumn(key="level", label="Level", no_wrap=True, ratio=1),
            ListColumn(key="step", label="Step", no_wrap=True, ratio=1),
            ListColumn(key="message", label="Message", ratio=6, overflow="fold"),
        ],
        display_items=[
            {**item, "timestamp": format_local_datetime(entry.timestamp)}
            for item, entry in zip(items, page.logs, strict=True)
        ],
        display_hints=[next_step_hint],
    )


def jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_secret_scopes(scopes: dict[str, Any] | None) -> str | None:
    if not scopes:
        return None

    parts: list[str] = []
    for name, raw_scope in sorted(scopes.items()):
        if not isinstance(name, str) or not isinstance(raw_scope, str):
            continue
        if raw_scope == "workspace":
            scope = "workspace"
        elif raw_scope == "user_override":
            scope = "personal, overrides workspace"
        elif raw_scope == "user":
            scope = "personal"
        else:
            scope = raw_scope
        parts.append(f"{name} ({scope})")
    return ", ".join(parts) if parts else None


def format_env_config(env_config: dict[str, Any] | None) -> str | None:
    if not env_config:
        return None
    parts = [
        f"{key}={jsonish(value)}"
        for key, value in sorted(env_config.items())
        if isinstance(key, str)
    ]
    return ", ".join(parts) if parts else None


def kv_section(title: str, rows: list[tuple[str, str]]) -> DetailSection | None:
    """Build a titled key/value section mirroring the main detail table.

    Values are passed through as plain text (never markup) so brackets and other
    Rich-significant characters render literally. Returns ``None`` when there is
    nothing to show so callers can append unconditionally.
    """
    if not rows:
        return None

    from rich import box
    from rich.table import Table
    from rich.text import Text

    table = Table(title=title, box=box.ROUNDED, show_header=False, title_justify="left")
    table.add_column("", style="cyan")
    table.add_column("")
    plain_lines = [f"{title}:"]
    for label, value in rows:
        table.add_row(label, Text(value))
        plain_lines.append(f"{label}: {value}")
    return DetailSection(rich=table, plain_lines=plain_lines)


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


def fetch_environment_secrets(
    client: Any,
    *,
    scope: str,
    credentials: Any,
    git_identity: str,
) -> list[Any] | None:
    """Best-effort fetch of every environment secret in ``scope``.

    *scope* is a wire value (``"workspace"`` / ``"user"``) or the literal
    ``"all"``. Returns the raw secret objects, or ``None`` on any failure
    (network, auth) so callers fall back to a best-effort path instead of
    blocking. Shared by the submit summary's scope partition and the
    ``secret delete`` existence pre-check.
    """
    try:

        def _fetch(limit: int, offset: int) -> Any:
            return client.list_environment_secrets(
                limit=limit,
                offset=offset,
                scope=scope,
                credentials=credentials,
                git_identity=git_identity,
            )

        secrets, _ = fetch_all_pages(_fetch, items_attr="environment_secrets")
        return secrets
    except Exception:
        return None


def paginated_fetch(
    fetch_fn: Callable[[int, int], Any],
    *,
    items_attr: str,
    limit: int,
    fetch_all: bool,
) -> tuple[list[Any], int, bool, int | None]:
    """Resolve a list endpoint's ``--all`` / ``--limit`` branch uniformly.

    *fetch_fn(limit, offset)* returns a page object exposing ``total_count``,
    ``has_more``, ``next_offset`` (int | None), and an attribute named
    *items_attr* (a list). The callable may be a closure with side effects
    (e.g. capturing a page-level ``platform_url``); it is invoked on every
    branch, so those side effects still fire.

    When *fetch_all* is True every page is walked via :func:`fetch_all_pages`
    and the result reports ``has_more=False`` / ``next_offset=None`` — a fully
    drained list has no continuation cursor. Otherwise a single page is fetched
    and its server-provided cursor fields are passed through verbatim.

    Returns ``(items, total_count, has_more, next_offset)``. ``next_offset``
    feeds ``ListResult.next_offset``, a required field of the stable ``--json``
    ``schema_version:1`` envelope, so it must never be dropped.
    """
    if fetch_all:
        items, total_count = fetch_all_pages(fetch_fn, items_attr=items_attr)
        return items, total_count, False, None
    page = fetch_fn(limit, 0)
    return getattr(page, items_attr), page.total_count, page.has_more, page.next_offset


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
