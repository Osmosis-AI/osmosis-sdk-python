from __future__ import annotations

import webbrowser
from typing import Any

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import Choice, Separator, confirm, is_interactive, select
from osmosis_ai.platform.auth import (
    PlatformAPIError,
    get_active_workspace,
    get_all_workspaces,
    set_active_workspace,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    clear_default_project,
    get_default_project,
    set_default_project,
)

from .constants import BACK, MSG_NOT_LOGGED_IN
from .utils import (
    format_dataset_status,
    format_processing_step,
    format_run_status,
    format_size,
)

app: typer.Typer = typer.Typer(help="Switch workspace and default project.")


def _validate_default_project(
    ws_name: str | None, default_project: dict | None
) -> dict | None:
    """Check the default project still exists; clear it if stale."""
    if not ws_name or not default_project:
        return default_project

    from .project import _get_cached_projects

    project_id = default_project.get("project_id")
    if not project_id:
        return default_project

    projects = _get_cached_projects(workspace_name=ws_name, max_age=0)
    if not projects:
        # If refresh failed or workspace has no projects, we can't confirm
        # whether the default still exists. Keep it to avoid clearing a valid
        # default due to a transient network issue. The browse functions
        # (_browse_runs, etc.) handle actual 404s via _handle_stale_project.
        return default_project

    for p in projects:
        if p.get("id") == project_id:
            return default_project

    # Project not found in the list — stale default
    clear_default_project(ws_name)
    console.print(
        f"Default project '{default_project.get('project_name', project_id)}' "
        "no longer exists. Please select a new project.",
        style="yellow",
    )
    console.print()
    return None


def _show_context(ws_name: str | None, default_project: dict | None) -> None:
    """Display current workspace/project context."""
    if not ws_name:
        console.print(console.format_styled("No workspace selected.", "dim"))
        console.print()
        return

    project_name = (
        default_project["project_name"]
        if default_project
        else console.format_styled("(no project selected)", "dim")
    )
    console.print(
        f"{console.format_styled('Current:', 'bold')} "
        f"{console.format_styled(ws_name, 'cyan')} / {project_name}"
    )
    url = f"{PLATFORM_URL}/{ws_name}"
    if default_project:
        url += f"/{default_project['project_name']}"
    console.print(
        f"{console.format_styled('URL:', 'bold')}     "
        f"{console.format_styled(url, 'dim')}"
    )
    console.print()


def _main_menu(has_project: bool) -> str | None:
    """Show main menu and return the selected action."""
    choices: list[str | Choice | Separator] = [
        Choice("Switch workspace / project", value="switch"),
    ]
    if has_project:
        choices.extend(
            [
                Choice("Browse training runs", value="runs"),
                Choice("Browse models", value="models"),
                Choice("Browse datasets", value="datasets"),
                Choice("View project info", value="info"),
                Choice("Open in browser", value="browser"),
            ]
        )
    choices.append(Separator())
    choices.append(Choice("Exit", value="exit"))

    console.separator()
    return select("Select an action", choices=choices)


def _select_workspace(
    workspaces: list[tuple],
    active_ws: str | None,
) -> str | None:
    """Prompt the user to select a workspace. Returns BACK or None to go back."""
    choices = []
    for name, creds, _is_active in workspaces:
        marker = " (current)" if name == active_ws else ""
        expired = " [expired]" if creds.is_expired() else ""
        title = f"{name}{marker}{expired}"
        choices.append(Choice(title, value=name))
    choices.append(Separator())
    choices.append(Choice("Back", value=BACK))

    console.separator()
    return select("Select workspace:", choices=choices)


def _select_project(ws_name: str) -> dict | str | None:
    """Prompt the user to select a default project.

    Returns:
        dict: Selected project with 'id' and 'project_name'.
        BACK: User chose to go back to workspace selection.
        None: User cancelled or skipped.
    """
    from .project import select_project_interactive

    default = get_default_project(ws_name)
    current_id = default.get("project_id") if default else None

    result = select_project_interactive(
        ws_name, current_project_id=current_id, allow_back=True
    )

    if result == BACK:
        return BACK
    if result is None:
        return None
    return result


def _switch_context(
    workspaces: list[tuple], active_ws: str | None
) -> tuple[str, dict | None] | None:
    """Run the workspace -> project -> confirm flow.

    Each step has a Back option to return to the previous step.
    Back from workspace selection returns to the main menu.

    Returns (ws_name, project_dict) on success, or None if backed out to main menu.
    """
    step = "workspace"
    ws_name = None
    result = None
    while True:
        if step == "workspace":
            ws_name = _select_workspace(workspaces, active_ws)
            if ws_name is None or ws_name == BACK:
                return None
            step = "project"

        elif step == "project":
            assert ws_name is not None
            result = _select_project(ws_name)
            if result == BACK or result is None:
                step = "workspace"
                continue
            assert isinstance(result, dict)
            step = "confirm"

        elif step == "confirm":
            assert ws_name is not None
            assert isinstance(result, dict)
            ok = confirm(f"Set context to {ws_name} / {result['project_name']}?")
            if ok is None or not ok:
                step = "project"
                continue

            # Apply changes
            if ws_name != active_ws:
                set_active_workspace(ws_name)

            set_default_project(ws_name, result["id"], result["project_name"])
            console.print(
                f"{console.format_styled('Switched to:', 'bold')} "
                f"{console.format_styled(ws_name, 'cyan')} / "
                f"{console.format_styled(result['project_name'], 'cyan')}"
            )
            return ws_name, {
                "project_name": result["project_name"],
                "project_id": result["id"],
            }


def _handle_stale_project(ws_name: str, project: dict) -> bool:
    """Handle a 404 by clearing the stale default project. Always returns False."""
    project_id = project.get("project_id")
    clear_default_project(ws_name)
    console.print_error(
        f"Project '{project.get('project_name', project_id)}' "
        "no longer exists. Default project has been cleared."
    )
    return False


def _browse_datasets(ws_name: str, project: dict) -> bool:
    """List datasets and allow selecting one for details.

    Returns False if the project was found to be stale (404).
    """
    from osmosis_ai.platform.api.client import OsmosisClient

    from .project import _get_workspace_credentials

    client = OsmosisClient()
    project_id: str = project["project_id"]
    try:
        credentials = _get_workspace_credentials(ws_name)
        result = client.list_datasets(project_id, credentials=credentials)
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load datasets: {e}")
        return True

    if not result.datasets:
        console.print("No datasets found.", style="dim")
        console.print()
        return True

    while True:
        choices: list[str | Choice | Separator] = []
        for d in result.datasets:
            status_str = format_dataset_status(d, for_prompt=True)
            label = f"{d.file_name} ({format_size(d.file_size)}) {status_str}"
            choices.append(Choice(label, value=d))
        choices.append(Separator())
        choices.append(Choice("Back", value=BACK))

        console.separator()
        selected = select(
            f"Datasets ({result.total_count}):",
            choices=choices,
        )

        if selected is None or selected == BACK:
            return True

        # Show dataset detail
        _show_dataset_detail(selected, ws_name, project)


def _show_dataset_detail(ds: Any, ws_name: str, project: dict) -> None:
    """Display detailed info for a single dataset."""
    rows = [
        ("File", ds.file_name),
        ("ID", ds.id),
        ("Size", format_size(ds.file_size)),
        ("Status", ds.status),
    ]
    step = format_processing_step(ds)
    if step:
        rows.append(("Step", step))
    if ds.error:
        rows.append(("Error", ds.error))
    if ds.created_at:
        rows.append(("Created", ds.created_at[:10]))

    url = f"{PLATFORM_URL}/{ws_name}/{project['project_name']}/training-data/{ds.id}"
    rows.append(("URL", url))

    console.table(rows, title="Dataset Detail")
    console.print()


def _browse_runs(ws_name: str, project: dict) -> bool:
    """List training runs and allow selecting one for details.

    Returns False if the project was found to be stale (404).
    """
    from osmosis_ai.platform.api.client import OsmosisClient

    from .project import _get_workspace_credentials

    client = OsmosisClient()
    project_id: str = project["project_id"]
    try:
        credentials = _get_workspace_credentials(ws_name)
        result = client.list_training_runs(project_id, credentials=credentials)
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load training runs: {e}")
        return True

    if not result.training_runs:
        console.print("No training runs found.", style="dim")
        console.print()
        return True

    while True:
        choices: list[str | Choice | Separator] = []
        for r in result.training_runs:
            status_str = format_run_status(r, for_prompt=True)
            name = r.name or "(unnamed)"
            model = r.model_name or ""
            label = f"{name}  {status_str}  {model}"
            choices.append(Choice(label, value=r))
        choices.append(Separator())
        choices.append(Choice("Back", value=BACK))

        console.separator()
        selected = select(
            f"Training Runs ({result.total_count}):",
            choices=choices,
        )

        if selected is None or selected == BACK:
            return True

        # Show run detail
        _show_run_detail(selected, ws_name, project)


def _show_run_detail(r: Any, ws_name: str, project: dict) -> None:
    """Display detailed info for a single training run."""
    rows = [
        ("Name", r.name or "(unnamed)"),
        ("ID", r.id),
        ("Status", r.status),
    ]
    step = format_processing_step(r)
    if step:
        rows.append(("Step", step))
    if r.model_name:
        rows.append(("Model", r.model_name))
    if r.eval_accuracy is not None:
        rows.append(("Accuracy", f"{r.eval_accuracy:.4f}"))
    if r.reward_increase_delta is not None:
        rows.append(("Reward Delta", f"{r.reward_increase_delta:+.4f}"))
    if r.error_message:
        rows.append(("Error", r.error_message))
    if r.creator_name:
        rows.append(("Creator", r.creator_name))
    if r.created_at:
        rows.append(("Created", r.created_at[:10]))

    url = f"{PLATFORM_URL}/{ws_name}/{project['project_name']}/training/{r.id}"
    rows.append(("URL", url))

    console.table(rows, title="Training Run")
    console.print()


def _browse_models(ws_name: str, project: dict) -> bool:
    """List models and allow selecting one for details.

    Returns False if the project was found to be stale (404).
    """
    from osmosis_ai.platform.api.client import OsmosisClient

    from .project import _get_workspace_credentials

    client = OsmosisClient()
    project_id: str = project["project_id"]
    try:
        credentials = _get_workspace_credentials(ws_name)
        result = client.list_models(project_id, credentials=credentials)
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load models: {e}")
        return True

    if not result.models:
        console.print("No models found.", style="dim")
        console.print()
        return True

    while True:
        choices: list[str | Choice | Separator] = []
        for m in result.models:
            base = f"  ({m.base_model})" if m.base_model else ""
            label = f"{m.model_name}  [{m.status}]{base}"
            choices.append(Choice(label, value=m))
        choices.append(Separator())
        choices.append(Choice("Back", value=BACK))

        console.separator()
        selected = select(
            f"Models ({result.total_count}):",
            choices=choices,
        )

        if selected is None or selected == BACK:
            return True

        # Show model detail
        _show_model_detail(selected)


def _show_model_detail(m: Any) -> None:
    """Display detailed info for a single model."""
    rows = [
        ("Model", m.model_name),
        ("ID", m.id),
        ("Status", m.status),
    ]
    if m.base_model:
        rows.append(("Base Model", m.base_model))
    if m.description:
        rows.append(("Description", m.description))
    if m.created_at:
        rows.append(("Created", m.created_at[:10]))

    console.table(rows, title="Model Detail")
    console.print()


def _show_project_info(ws_name: str, project: dict) -> bool:
    """Fetch and display project details.

    Returns False if the project was found to be stale (404).
    """
    from osmosis_ai.platform.api.client import OsmosisClient

    from .project import _get_workspace_credentials

    client = OsmosisClient()
    project_id: str = project["project_id"]
    try:
        credentials = _get_workspace_credentials(ws_name)
        detail = client.get_project(project_id, credentials=credentials)
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load project info: {e}")
        return True

    url = f"{PLATFORM_URL}/{ws_name}/{detail.project_name}"
    rows = [
        ("Project", detail.project_name),
        ("ID", detail.id),
        ("Role", detail.role),
        ("Datasets", str(detail.dataset_count)),
        ("Training Runs", str(detail.training_run_count)),
        ("Base Models", str(detail.base_model_count)),
        ("Output Models", str(detail.output_model_count)),
    ]
    if detail.created_at:
        rows.append(("Created", detail.created_at[:10]))
    rows.append(("URL", url))

    console.table(rows, title="Project Info")
    console.print()
    return True


def _open_in_browser(ws_name: str, project: dict) -> None:
    """Open the current workspace/project URL in the default browser."""
    url = f"{PLATFORM_URL}/{ws_name}/{project['project_name']}"
    console.print(f"Opening {console.format_styled(url, 'dim')} ...")
    webbrowser.open(url)
    console.print()


@app.callback(invoke_without_command=True)
def workspace() -> None:
    """Switch workspace and default project."""
    workspaces = get_all_workspaces()

    if not workspaces:
        raise CLIError(MSG_NOT_LOGGED_IN)

    active_ws = get_active_workspace()
    ws_name = active_ws
    default_project = get_default_project(active_ws) if active_ws else None

    # Validate the default project still exists
    default_project = _validate_default_project(ws_name, default_project)

    # Show current context
    _show_context(ws_name, default_project)

    # Non-interactive: just show current context
    if not is_interactive():
        return

    # Interactive main menu loop
    while True:
        action = _main_menu(has_project=bool(default_project))

        if action is None or action == "exit":
            return
        elif action == "switch":
            result = _switch_context(workspaces, active_ws)
            if result:
                ws_name, default_project = result
                active_ws = ws_name
                console.print()
                _show_context(ws_name, default_project)
        elif action == "runs":
            assert ws_name is not None and default_project is not None
            if not _browse_runs(ws_name, default_project):
                default_project = None  # Already cleared by _handle_stale_project
                _show_context(ws_name, default_project)
        elif action == "models":
            assert ws_name is not None and default_project is not None
            if not _browse_models(ws_name, default_project):
                default_project = None  # Already cleared by _handle_stale_project
                _show_context(ws_name, default_project)
        elif action == "datasets":
            assert ws_name is not None and default_project is not None
            if not _browse_datasets(ws_name, default_project):
                default_project = None  # Already cleared by _handle_stale_project
                _show_context(ws_name, default_project)
        elif action == "info":
            assert ws_name is not None and default_project is not None
            if not _show_project_info(ws_name, default_project):
                default_project = None  # Already cleared by _handle_stale_project
                _show_context(ws_name, default_project)
        elif action == "browser":
            assert ws_name is not None and default_project is not None
            _open_in_browser(ws_name, default_project)
