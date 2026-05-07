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
    load_credentials,
    platform_request,
)
from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

from .constants import BACK, DEFAULT_VISIBLE_CHOICES
from .utils import (
    build_dataset_detail_rows,
    build_run_detail_rows,
    format_dataset_status,
    format_date,
    format_run_status,
    format_size,
    platform_call,
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
    """Display linked project workspace context."""
    if not ws_name:
        console.print(
            console.format_styled(
                "This project is not linked to an Osmosis workspace for the "
                "current platform.",
                "dim",
            )
        )
        console.print()
        return

    console.print(
        f"{console.format_styled('Current:', 'bold')} "
        f"{console.format_styled(ws_name, 'cyan')}"
    )
    url = platform_entity_url(ws_name)
    console.print_url("URL:     ", url, style="dim")
    console.print()


def _main_menu(has_workspace: bool) -> str | None:
    """Show main menu and return the selected action."""
    choices: list[str | Choice | Separator] = [
        Choice("Browse workspace", value="browse"),
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
            # Read-only menu fetch: a transient 401 must not wipe local
            # credentials/workspace state out from under the user mid-menu.
            result = platform_request(
                "/api/cli/workspaces",
                require_workspace=False,
                cleanup_on_401=False,
            )
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


def _upload_dataset_interactive(
    ws_name: str,
    workspace_id: str,
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

    console.print(
        f"  File: {console.escape(file_path.name)} ({format_size(file_size)})"
    )
    ok = confirm("Upload to this workspace?", default=True)
    if not ok:
        return False

    try:
        dataset = _perform_upload(
            file_path=file_path,
            ext=ext,
            file_size=file_size,
            workspace_id=workspace_id,
            credentials=credentials,
        )
    except CLIError as e:
        console.print_error(str(e))
        return False

    console.print(
        f"Upload complete. Dataset: {console.escape(dataset.file_name)}",
        style="green",
        highlight=False,
    )
    url = platform_entity_url(ws_name, "datasets", dataset.id)
    console.print_url("Check status at: ", url)
    return True


_UPLOAD = "__upload__"


def _browse_datasets(ws_name: str, workspace_id: str) -> bool:
    """List datasets and allow selecting one for details or uploading."""
    from osmosis_ai.platform.api.client import OsmosisClient

    from .utils import require_credentials

    client = OsmosisClient()

    try:
        credentials = require_credentials()
        with console.spinner("Loading datasets..."):
            result = client.list_datasets(
                credentials=credentials,
                workspace_id=workspace_id,
            )
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
            uploaded = _upload_dataset_interactive(
                ws_name,
                workspace_id,
                credentials,
            )
            if uploaded:
                # Refresh the list after successful upload
                with contextlib.suppress(PlatformAPIError):
                    result = platform_call(
                        "Refreshing datasets...",
                        lambda: client.list_datasets(
                            credentials=credentials,
                            workspace_id=workspace_id,
                        ),
                        output_console=console,
                    )
            continue

        _show_dataset_detail(selected, ws_name)


def _show_dataset_detail(ds: Any, ws_name: str) -> None:
    """Display detailed info for a single dataset."""
    rows = build_dataset_detail_rows(ds)
    if ds.created_at:
        rows.append(("Created", format_date(ds.created_at)))
    url = platform_entity_url(ws_name, "datasets", ds.id)

    console.table(rows, title="Dataset Detail")
    _print_platform_link(url)
    console.print()


def _browse_runs(ws_name: str, workspace_id: str) -> None:
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
            result = client.list_training_runs(
                credentials=credentials,
                workspace_id=workspace_id,
            )
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

    console.table(rows, title="Training Run")
    _print_platform_link(url)
    console.print()


def _browse_models(ws_name: str, workspace_id: str) -> None:
    """List models and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    try:
        credentials = require_credentials()
        with console.spinner("Loading models..."):
            result = client.list_base_models(
                credentials=credentials,
                workspace_id=workspace_id,
            )
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
    rows: list[tuple[Any, Any]] = [
        ("Model", console.format_text(m.model_name)),
        ("ID", console.format_text(m.id)),
        ("Status", console.format_text(m.status)),
    ]
    if m.base_model:
        rows.append(("Base Model", console.format_text(m.base_model)))
    if m.description:
        rows.append(("Description", console.format_text(m.description)))
    if m.creator_name:
        rows.append(("Creator", console.format_text(m.creator_name)))
    if m.created_at:
        rows.append(("Created", format_date(m.created_at)))
    url = platform_entity_url(ws_name, "models", m.id)

    console.table(rows, title="Model Detail")
    _print_platform_link(url)
    console.print()


def _print_platform_link(url: str) -> None:
    """Print a copyable platform URL outside tables so Rich won't truncate it."""
    console.print_url("View on platform: ", url, style="cyan")


def _open_in_browser(ws_name: str) -> None:
    """Open the workspace URL in the default browser."""
    url = platform_entity_url(ws_name)
    console.print_url("Opening ", url, style="dim")
    webbrowser.open(url)
    console.print()


def _workspace_ui_error() -> CLIError:
    return CLIError(
        "Interactive workspace UI is unavailable in this mode. "
        "Use 'osmosis workspace list' or 'osmosis workspace create <name>'.",
        code="INTERACTIVE_REQUIRED",
    )


def _workspace_summary(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ws["id"],
        "name": ws["name"],
    }


def list_workspaces() -> Any:
    """List all workspaces (non-interactive)."""
    from osmosis_ai.cli.output import (
        ListColumn,
        ListResult,
        OutputFormat,
        get_output_context,
        serialize_workspace,
    )

    output = get_output_context()
    credentials = require_credentials()
    result = platform_call(
        "Loading workspaces...",
        lambda: platform_request(
            "/api/cli/workspaces",
            require_workspace=False,
            credentials=credentials,
            cleanup_on_401=False,
        ),
        output_console=console,
    )
    workspaces = result.get("workspaces", [])

    if not workspaces:
        if output.format is OutputFormat.rich:
            console.print("No workspaces found.")
            return None
        return ListResult(
            title="Workspaces",
            items=[],
            total_count=0,
            has_more=False,
            next_offset=None,
            columns=[ListColumn(key="name", label="Name")],
        )

    if output.format is not OutputFormat.rich:
        items = []
        for ws in workspaces:
            item = serialize_workspace(_workspace_summary(ws))
            if "has_subscription" in ws:
                item["has_subscription"] = bool(ws.get("has_subscription"))
            items.append(item)
        return ListResult(
            title="Workspaces",
            items=items,
            total_count=len(items),
            has_more=False,
            next_offset=None,
            columns=[
                ListColumn(key="name", label="Name"),
                ListColumn(key="id", label="ID", plain=False),
            ],
        )

    console.print(f"Workspaces ({len(workspaces)}):", style="bold")
    for ws in workspaces:
        raw_name = ws.get("name", "")
        name = console.escape(raw_name)
        sub_label = "active" if ws.get("has_subscription") else "no subscription"
        console.print(f"  {name}  {console.escape(f'[{sub_label}]')}")
    return None


def create_workspace(name: str, timezone: str) -> Any:
    """Create a new workspace."""
    from osmosis_ai.cli.output import (
        OperationResult,
        OutputFormat,
        get_output_context,
        serialize_workspace,
    )
    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    credentials = require_credentials()

    client = OsmosisClient()
    result = platform_call(
        "Creating workspace...",
        lambda: client.create_workspace(name, timezone, credentials=credentials),
        output_console=console,
    )

    resource = serialize_workspace(_workspace_summary(result))
    resource["timezone"] = timezone

    if output.format is OutputFormat.rich:
        console.print(
            f"Workspace '{console.escape(result['name'])}' created.",
            style="green",
            highlight=False,
        )
        return None

    return OperationResult(
        operation="workspace.create",
        status="success",
        resource=resource,
        message=f"Workspace '{result['name']}' created.",
        display_next_steps=[
            f"Link from a project with: osmosis project link --workspace {result['id']}"
        ],
    )


def _require_delete_confirmation(*, workspace_name: str, yes: bool) -> None:
    from osmosis_ai.cli.output import OutputFormat, get_output_context
    from osmosis_ai.cli.prompts import require_confirmation

    if yes:
        return
    output = get_output_context()
    if output.format is not OutputFormat.rich or not output.interactive:
        raise CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )
    require_confirmation(
        f"Delete workspace '{workspace_name}'? "
        "This will stop all running processes and delete all datasets and training "
        "runs. This cannot be undone.",
        yes=False,
    )


def delete_workspace(name: str, *, yes: bool = False) -> Any:
    """Delete a workspace."""
    from osmosis_ai.cli.output import (
        OperationResult,
        OutputFormat,
        get_output_context,
        serialize_workspace,
    )
    from osmosis_ai.platform.api.client import OsmosisClient

    output = get_output_context()
    if not yes and (output.format is not OutputFormat.rich or not output.interactive):
        raise CLIError(
            "Use --yes to confirm in non-interactive mode.",
            code="INTERACTIVE_REQUIRED",
        )

    credentials = require_credentials()
    client = OsmosisClient()

    ws_data = platform_call(
        "Loading workspaces...",
        lambda: client.list_workspaces(credentials=credentials),
        output_console=console,
    )
    workspace = None
    for ws in ws_data.get("workspaces", []):
        if ws.get("name", "").lower() == name.lower():
            workspace = ws
            break

    if not workspace:
        raise CLIError(f"Workspace '{name}' not found.", code="NOT_FOUND")

    status = platform_call(
        "Checking workspace deletion safety...",
        lambda: client.get_workspace_deletion_status(
            workspace["id"], credentials=credentials
        ),
        output_console=console,
    )

    if not status.is_owner:
        raise CLIError("Only workspace owners can delete a workspace.")

    if status.is_last_workspace:
        raise CLIError("Cannot delete your only workspace.")

    if output.format is OutputFormat.rich and status.has_running_processes:
        console.print(
            "This workspace has running processes:",
            style="yellow",
        )
        processes: list[str] = []
        if not status.feature_pipelines.valid:
            processes.append(f"{status.feature_pipelines.count} pipeline(s)")
        if not status.training_runs.valid:
            processes.append(f"{status.training_runs.count} training run(s)")
        if not status.models.valid:
            processes.append(f"{status.models.count} active model(s)")
        if processes:
            console.print(f"  {', '.join(processes)}")
        console.print()

    _require_delete_confirmation(workspace_name=workspace["name"], yes=yes)

    platform_call(
        "Deleting workspace...",
        lambda: client.delete_workspace(workspace["id"], credentials=credentials),
        output_console=console,
    )

    if output.format is OutputFormat.rich:
        console.print(
            f"Workspace '{console.escape(workspace['name'])}' deleted.",
            style="green",
            highlight=False,
        )
        return None

    return OperationResult(
        operation="workspace.delete",
        status="success",
        resource=serialize_workspace(_workspace_summary(workspace)),
        message=f"Workspace '{workspace['name']}' deleted.",
    )


@app.callback(invoke_without_command=True)
def workspace() -> None:
    """Manage workspace context."""
    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is not OutputFormat.rich:
        raise _workspace_ui_error()

    credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)

    ws_id = None
    ws_name = None
    with contextlib.suppress(CLIError):
        from osmosis_ai.platform.cli.workspace_context import (
            resolve_linked_workspace_context,
        )

        linked_workspace = resolve_linked_workspace_context()
        ws_id = linked_workspace.workspace_id
        ws_name = linked_workspace.workspace_name

    _show_context(ws_name)

    if not is_interactive():
        return

    while True:
        action = _main_menu(has_workspace=bool(ws_name))

        if action is None or action == "exit":
            return
        if action == "browse":
            selected = _select_workspace(ws_name)
            if isinstance(selected, tuple):
                ws_id, ws_name = selected
                console.print()
                _show_context(ws_name)
        elif action == "datasets":
            if ws_name is None or ws_id is None:
                continue
            _browse_datasets(ws_name, ws_id)
        elif action == "runs":
            if ws_name is None or ws_id is None:
                continue
            _browse_runs(ws_name, ws_id)
        elif action == "models":
            if ws_name is None or ws_id is None:
                continue
            _browse_models(ws_name, ws_id)
        elif action == "browser":
            if ws_name is None:
                continue
            _open_in_browser(ws_name)
