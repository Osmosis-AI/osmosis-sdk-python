from __future__ import annotations

import contextlib
import webbrowser
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import (
    Choice,
    Separator,
    confirm,
    is_interactive,
    select,
    select_list,
    text,
)
from osmosis_ai.platform.auth import (
    PlatformAPIError,
    ensure_active_workspace,
    load_credentials,
    platform_request,
)
from osmosis_ai.platform.auth.local_config import set_active_workspace
from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

from .constants import BACK, DEFAULT_VISIBLE_CHOICES
from .utils import (
    build_dataset_detail_rows,
    build_run_detail_rows,
    format_dataset_status,
    format_date,
    format_run_status,
    format_size,
    platform_entity_url,
    require_credentials,
)

app: typer.Typer = typer.Typer(help="Manage workspace context.")


def _clean_file_path(raw: str) -> str:
    """Normalize a file path from drag-and-drop input.

    Handles: surrounding whitespace, single/double quotes, backslash-escaped spaces.
    """
    path = raw.strip()
    if (path.startswith("'") and path.endswith("'")) or (
        path.startswith('"') and path.endswith('"')
    ):
        path = path[1:-1]
    path = path.replace("\\ ", " ")
    return path


def _show_context(ws_name: str | None) -> None:
    """Display current workspace context."""
    if not ws_name:
        console.print(console.format_styled("No workspace selected.", "dim"))
        console.print()
        return

    console.print(
        f"{console.format_styled('Current:', 'bold')} "
        f"{console.format_styled(ws_name, 'cyan')}"
    )
    url = platform_entity_url(ws_name)
    console.print(
        f"{console.format_styled('URL:', 'bold')}     "
        f"{console.format_styled(url, 'dim')}"
    )
    console.print()


def _main_menu(has_workspace: bool) -> str | None:
    """Show main menu and return the selected action."""
    choices: list[str | Choice | Separator] = [
        Choice("Change workspace", value="switch"),
    ]
    if has_workspace:
        choices.extend(
            [
                Choice("Datasets", value="datasets"),
                Choice("Training runs", value="runs"),
                Choice("Models", value="models"),
                Choice("Open in browser", value="browser"),
            ]
        )
    choices.append(Separator())
    choices.append(Choice("Exit", value="exit"))

    console.separator()
    return select("What would you like to do?", choices=choices)


def _select_workspace(
    active_ws_name: str | None,
) -> tuple[str, str] | str | None:
    """Prompt the user to select a workspace. Returns (ws_id, ws_name), BACK, or None."""
    try:
        with console.spinner("Loading workspaces..."):
            result = platform_request("/api/cli/workspaces", require_workspace=False)
        workspaces = result.get("workspaces", [])
    except PlatformAPIError as e:
        console.print_error(f"Failed to load workspaces: {e}")
        return BACK

    if not workspaces:
        console.print("No workspaces found.", style="dim")
        return BACK

    choices = []
    for ws in workspaces:
        name = ws.get("name", "")
        ws_id = ws.get("id", "")
        marker = " (current)" if name == active_ws_name else ""
        title = f"{name}{marker}"
        choices.append(Choice(title, value=(ws_id, name)))
    choices.append(Separator())
    choices.append(Choice("Back", value=BACK))

    console.separator()
    return select("Choose a workspace", choices=choices)


def _switch_context(active_ws_name: str | None) -> str | None:
    """Select a workspace, confirm, and set it active.

    Returns the new workspace name on success, or None if the user backs out.
    """
    selected = _select_workspace(active_ws_name)
    if selected is None or selected == BACK:
        return None
    ws_id, ws_name = selected
    ok = confirm(f"Switch to workspace {ws_name}?")
    if not ok:
        return None
    if ws_name != active_ws_name:
        set_active_workspace(ws_id, ws_name)
    console.print(
        f"{console.format_styled('Switched to:', 'bold')} "
        f"{console.format_styled(ws_name, 'cyan')}"
    )
    return ws_name


def _upload_dataset_interactive(
    ws_name: str,
    credentials: Any,
) -> bool:
    """Prompt for a file path and upload it as a dataset.

    Returns True on success, False on cancel/error.
    """
    from osmosis_ai.platform.cli.dataset import (
        _check_file_basics,
        _perform_upload,
        _validate_file,
    )

    raw = text("Drop a file here or enter path:")
    if not raw:
        return False

    file_str = _clean_file_path(raw)

    try:
        file_path, ext, file_size = _check_file_basics(file_str)
    except CLIError as e:
        console.print_error(str(e))
        return False

    errors = _validate_file(file_path, ext)
    if errors:
        console.print_error(
            "Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
        return False

    console.print(f"  File: {file_path.name} ({format_size(file_size)})")
    ok = confirm("Upload to this workspace?", default=True)
    if not ok:
        return False

    try:
        dataset = _perform_upload(
            file_path=file_path,
            ext=ext,
            file_size=file_size,
            credentials=credentials,
        )
    except CLIError as e:
        console.print_error(str(e))
        return False

    console.print(
        f"Upload complete. Dataset: {console.escape(dataset.file_name)}",
        style="green",
    )
    url = platform_entity_url(ws_name, "datasets", dataset.id)
    console.print(f"Check status at: {url}")
    return True


_UPLOAD = "__upload__"


def _browse_datasets(ws_name: str) -> bool:
    """List datasets and allow selecting one for details or uploading."""
    from osmosis_ai.platform.api.client import OsmosisClient

    from .utils import require_credentials

    client = OsmosisClient()

    try:
        credentials = require_credentials()
        with console.spinner("Loading datasets..."):
            result = client.list_datasets(credentials=credentials)
    except PlatformAPIError as e:
        console.print_error(f"Failed to load datasets: {e}")
        return True

    while True:
        items: list[str | Choice | Separator] = [
            Choice(
                f"{d.file_name} ({format_size(d.file_size)}) "
                f"{format_dataset_status(d, for_prompt=True)}",
                value=d,
            )
            for d in result.datasets
        ]

        console.separator()
        selected = select_list(
            f"Datasets ({result.total_count}):",
            items=items,
            actions=[
                Choice("Upload dataset", value=_UPLOAD),
                Choice("Back", value=BACK),
            ],
            max_visible=DEFAULT_VISIBLE_CHOICES,
        )

        if selected is None or selected == BACK:
            return True

        if selected == _UPLOAD:
            uploaded = _upload_dataset_interactive(ws_name, credentials)
            if uploaded:
                # Refresh the list after successful upload
                with contextlib.suppress(PlatformAPIError):
                    result = client.list_datasets(credentials=credentials)
            continue

        _show_dataset_detail(selected, ws_name)


def _show_dataset_detail(ds: Any, ws_name: str) -> None:
    """Display detailed info for a single dataset."""
    rows = build_dataset_detail_rows(ds)
    if ds.created_at:
        rows.append(("Created", format_date(ds.created_at)))
    url = platform_entity_url(ws_name, "datasets", ds.id)
    rows.append(("URL", url))

    console.table(rows, title="Dataset Detail")
    console.print()


def _browse_runs(ws_name: str) -> None:
    """List training runs and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    def _format(r: Any) -> str:
        status_str = format_run_status(r, for_prompt=True)
        name = r.name or "(unnamed)"
        model = r.model_name or ""
        return f"{name}  {status_str}  {model}"

    try:
        credentials = require_credentials()
        with console.spinner("Loading training runs..."):
            result = client.list_training_runs(credentials=credentials)
    except PlatformAPIError as e:
        console.print_error(f"Failed to load training runs: {e}")
        return

    items = result.training_runs
    if not items:
        console.print("No training runs found.", style="dim")
        console.print()
        return

    data_choices: list[str | Choice | Separator] = [
        Choice(_format(item), value=item) for item in items
    ]

    while True:
        console.separator()
        selected = select_list(
            f"Training Runs ({result.total_count}):",
            items=data_choices,
            actions=[Choice("Back", value=BACK)],
            max_visible=DEFAULT_VISIBLE_CHOICES,
        )

        if selected is None or selected == BACK:
            return

        _show_run_detail(selected, ws_name)


def _show_run_detail(r: Any, ws_name: str) -> None:
    """Display detailed info for a single training run."""
    rows = build_run_detail_rows(r)
    url = platform_entity_url(ws_name, "training", r.id)
    rows.append(("URL", url))

    console.table(rows, title="Training Run")
    console.print()


def _browse_models(ws_name: str) -> None:
    """List models and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    try:
        credentials = require_credentials()
        with console.spinner("Loading models..."):
            result = client.list_base_models(credentials=credentials)
    except PlatformAPIError as e:
        console.print_error(f"Failed to load models: {e}")
        return

    if not result.models:
        console.print("No models found.", style="dim")
        console.print()
        return

    model_items: list[str | Choice | Separator] = []
    for m in result.models:
        creator = f"  by {m.creator_name}" if m.creator_name else ""
        label = f"{m.model_name}  [{m.status}]{creator}"
        model_items.append(Choice(label, value=m))

    while True:
        console.separator()
        selected = select_list(
            f"Models ({result.total_count}):",
            items=model_items,
            actions=[Choice("Back", value=BACK)],
            max_visible=DEFAULT_VISIBLE_CHOICES,
        )

        if selected is None or selected == BACK:
            return

        _show_model_detail(selected, ws_name)


def _show_model_detail(m: Any, ws_name: str) -> None:
    """Display detailed info for a single model."""
    rows: list[tuple[str, str]] = [
        ("Model", m.model_name),
        ("ID", m.id),
        ("Status", m.status),
    ]
    if m.base_model:
        rows.append(("Base Model", m.base_model))
    if m.description:
        rows.append(("Description", m.description))
    if m.creator_name:
        rows.append(("Creator", m.creator_name))
    if m.created_at:
        rows.append(("Created", format_date(m.created_at)))
    url = platform_entity_url(ws_name, "models", m.id)
    rows.append(("URL", url))

    console.table(rows, title="Model Detail")
    console.print()


def _open_in_browser(ws_name: str) -> None:
    """Open the workspace URL in the default browser."""
    url = platform_entity_url(ws_name)
    console.print(f"Opening {console.format_styled(url, 'dim')} ...")
    webbrowser.open(url)
    console.print()


def list_workspaces() -> None:
    """List all workspaces (non-interactive)."""
    credentials = require_credentials()
    active_ws = ensure_active_workspace(credentials=credentials)
    result = platform_request(
        "/api/cli/workspaces", require_workspace=False, credentials=credentials
    )
    workspaces = result.get("workspaces", [])

    if not workspaces:
        console.print("No workspaces found.")
        return

    active_name = active_ws["name"] if active_ws else None

    console.print(f"Workspaces ({len(workspaces)}):", style="bold")
    for ws in workspaces:
        name = ws.get("name", "")
        marker = " (current)" if name == active_name else ""
        sub_label = "active" if ws.get("has_subscription") else "no subscription"
        console.print(f"  {name}{marker}  [{sub_label}]")


def switch_workspace(workspace: str) -> None:
    """Switch to a different workspace."""
    credentials = require_credentials()
    result = platform_request(
        "/api/cli/workspaces", require_workspace=False, credentials=credentials
    )
    workspaces = result.get("workspaces", [])

    target = workspace.lower()
    matched = [ws for ws in workspaces if ws.get("name", "").lower() == target]
    if not matched:
        available = ", ".join(ws.get("name", "") for ws in workspaces) or "(none)"
        raise CLIError(
            f"Workspace '{workspace}' not found.\n  Available workspaces: {available}"
        )

    ws = matched[0]
    ws_id = ws["id"]
    ws_name = ws["name"]

    set_active_workspace(ws_id, ws_name)
    console.print(f"Switched to workspace: {console.format_styled(ws_name, 'cyan')}")


@app.callback(invoke_without_command=True)
def workspace() -> None:
    """Manage workspace context."""
    credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)

    active_ws = ensure_active_workspace(credentials=credentials)
    ws_name = active_ws["name"] if active_ws else None

    _show_context(ws_name)

    if not is_interactive():
        return

    while True:
        action = _main_menu(has_workspace=bool(ws_name))

        if action is None or action == "exit":
            return
        if action == "switch":
            new_ws = _switch_context(ws_name)
            if new_ws is not None:
                ws_name = new_ws
                console.print()
                _show_context(ws_name)
        elif action == "datasets":
            if ws_name is None:
                continue
            _browse_datasets(ws_name)
        elif action == "runs":
            if ws_name is None:
                continue
            _browse_runs(ws_name)
        elif action == "models":
            if ws_name is None:
                continue
            _browse_models(ws_name)
        elif action == "browser":
            if ws_name is None:
                continue
            _open_in_browser(ws_name)
