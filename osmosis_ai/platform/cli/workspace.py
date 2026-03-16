from __future__ import annotations

import webbrowser
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

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
)
from osmosis_ai.platform.auth import (
    PlatformAPIError,
    get_all_workspaces,
    set_active_workspace,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.local_config import (
    clear_default_project,
    get_default_project,
    set_default_project,
)

from .constants import BACK, DEFAULT_VISIBLE_CHOICES, MSG_NOT_LOGGED_IN
from .utils import (
    build_dataset_detail_rows,
    build_run_detail_rows,
    format_dataset_status,
    format_date,
    format_run_status,
    format_size,
    platform_entity_url,
)

app: typer.Typer = typer.Typer(help="Manage workspace and project context.")


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

    projects = _get_cached_projects(workspace_name=ws_name)
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
    if default_project:
        url = platform_entity_url(ws_name, default_project["project_name"])
    else:
        url = f"{PLATFORM_URL}/{ws_name}"
    console.print(
        f"{console.format_styled('URL:', 'bold')}     "
        f"{console.format_styled(url, 'dim')}"
    )
    console.print()


def _main_menu(has_project: bool) -> str | None:
    """Show main menu and return the selected action."""
    choices: list[str | Choice | Separator] = [
        Choice("Change workspace or project", value="switch"),
    ]
    if has_project:
        choices.extend(
            [
                Choice("Training runs", value="runs"),
                Choice("Models", value="models"),
                Choice("Datasets", value="datasets"),
                Choice("Project details", value="info"),
                Choice("Open in browser", value="browser"),
            ]
        )
    choices.append(Separator())
    choices.append(Choice("Exit", value="exit"))

    console.separator()
    return select("What would you like to do?", choices=choices)


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
    return select("Choose a workspace", choices=choices)


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
            if ws_name is None:
                step = "workspace"
                continue
            result = _select_project(ws_name)
            if result == BACK:
                step = "workspace"
                continue
            if result is None:
                # Allow switching workspace without selecting a project
                # (e.g. empty workspaces with no projects yet)
                ok = confirm(f"Switch to {ws_name}? (no project will be selected)")
                if not ok:
                    step = "workspace"
                    continue
                if ws_name != active_ws:
                    set_active_workspace(ws_name)
                clear_default_project(ws_name)
                console.print(
                    f"{console.format_styled('Switched to:', 'bold')} "
                    f"{console.format_styled(ws_name, 'cyan')}"
                )
                return ws_name, None
            if not isinstance(result, dict):
                step = "workspace"
                continue
            step = "confirm"

        elif step == "confirm":
            if ws_name is None or not isinstance(result, dict):
                step = "workspace"
                continue
            ok = confirm(f"Switch to {ws_name} / {result['project_name']}?")
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


def _browse_entities(
    ws_name: str,
    project: dict,
    *,
    fetch: Callable,
    extract_items: Callable[[Any], Sequence],
    format_choice: Callable[[Any], str],
    show_detail: Callable[[Any], None],
    title: str,
) -> bool:
    """Generic browse helper for datasets, runs, and models.

    Args:
        ws_name: Active workspace name.
        project: Default project dict with 'project_id' and 'project_name'.
        fetch: Callable(project_id, credentials=...) that returns an API result.
        extract_items: Extracts the entity list from the API result.
        format_choice: Formats an entity into a Choice label string.
        show_detail: Displays detailed info for a selected entity.
        title: Display title (e.g. "Datasets", "Training Runs", "Models").

    Returns False if the project was found to be stale (404).
    """
    from .project import _get_workspace_credentials

    project_id: str = project["project_id"]
    try:
        credentials = _get_workspace_credentials(ws_name)
        result = fetch(project_id, credentials=credentials)
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load {title.lower()}: {e}")
        return True

    items = extract_items(result)
    if not items:
        console.print(f"No {title.lower()} found.", style="dim")
        console.print()
        return True

    data_choices: list[str | Choice | Separator] = [
        Choice(format_choice(item), value=item) for item in items
    ]

    while True:
        console.separator()
        selected = select_list(
            f"{title} ({result.total_count}):",
            items=data_choices,
            actions=[Choice("Back", value=BACK)],
            max_visible=DEFAULT_VISIBLE_CHOICES,
        )

        if selected is None or selected == BACK:
            return True

        show_detail(selected)


def _browse_datasets(ws_name: str, project: dict) -> bool:
    """List datasets and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    def _format(d: Any) -> str:
        status_str = format_dataset_status(d, for_prompt=True)
        return f"{d.file_name} ({format_size(d.file_size)}) {status_str}"

    return _browse_entities(
        ws_name,
        project,
        fetch=client.list_datasets,
        extract_items=lambda r: r.datasets,
        format_choice=_format,
        show_detail=lambda d: _show_dataset_detail(d, ws_name, project),
        title="Datasets",
    )


def _show_dataset_detail(ds: Any, ws_name: str, project: dict) -> None:
    """Display detailed info for a single dataset."""
    rows = build_dataset_detail_rows(ds)
    if ds.created_at:
        rows.append(("Created", format_date(ds.created_at)))
    url = platform_entity_url(ws_name, project["project_name"], "training-data", ds.id)
    rows.append(("URL", url))

    console.table(rows, title="Dataset Detail")
    console.print()


def _browse_runs(ws_name: str, project: dict) -> bool:
    """List training runs and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    def _format(r: Any) -> str:
        status_str = format_run_status(r, for_prompt=True)
        name = r.name or "(unnamed)"
        model = r.model_name or ""
        return f"{name}  {status_str}  {model}"

    return _browse_entities(
        ws_name,
        project,
        fetch=client.list_training_runs,
        extract_items=lambda r: r.training_runs,
        format_choice=_format,
        show_detail=lambda r: _show_run_detail(r, ws_name, project),
        title="Training Runs",
    )


def _show_run_detail(r: Any, ws_name: str, project: dict) -> None:
    """Display detailed info for a single training run."""
    rows = build_run_detail_rows(r)
    url = platform_entity_url(ws_name, project["project_name"], "training", r.id)
    rows.append(("URL", url))

    console.table(rows, title="Training Run")
    console.print()


def _browse_models(ws_name: str, project: dict) -> bool:
    """List base and output models and allow selecting one for details."""
    from osmosis_ai.platform.api.client import OsmosisClient

    from .project import _get_workspace_credentials

    client = OsmosisClient()
    project_id: str = project["project_id"]

    try:
        credentials = _get_workspace_credentials(ws_name)
        base_result, output_result = client.fetch_all_models(
            project_id, credentials=credentials
        )
    except PlatformAPIError as e:
        if e.status_code == 404:
            return _handle_stale_project(ws_name, project)
        console.print_error(f"Failed to load models: {e}")
        return True

    if not base_result.models and not output_result.models:
        console.print("No models found.", style="dim")
        console.print()
        return True

    model_items: list[str | Choice | Separator] = []
    if base_result.models:
        model_items.append(Separator("── Base Models ──"))
        for m in base_result.models:
            creator = f"  by {m.creator_name}" if m.creator_name else ""
            label = f"{m.model_name}  [{m.status}]{creator}"
            model_items.append(Choice(label, value=("base", m)))
    if output_result.models:
        model_items.append(Separator("── Output Models ──"))
        for m in output_result.models:
            run = f"  from {m.training_run_name}" if m.training_run_name else ""
            label = f"{m.model_name}  [{m.status}]{run}"
            model_items.append(Choice(label, value=("output", m)))

    total = base_result.total_count + output_result.total_count
    while True:
        console.separator()
        selected = select_list(
            f"Models ({total}):",
            items=model_items,
            actions=[Choice("Back", value=BACK)],
            max_visible=DEFAULT_VISIBLE_CHOICES,
        )

        if selected is None or selected == BACK:
            return True

        kind, model = selected
        _show_model_detail(
            cast(Literal["base", "output"], kind), model, ws_name, project
        )


def _show_model_detail(
    kind: Literal["base", "output"], m: Any, ws_name: str, project: dict
) -> None:
    """Display detailed info for a single model."""
    rows: list[tuple[str, str]] = [
        ("Model", m.model_name),
        ("ID", m.id),
        ("Type", "Base Model" if kind == "base" else "Output Model"),
        ("Status", m.status),
    ]
    if m.base_model:
        rows.append(("Base Model", m.base_model))
    if m.description:
        rows.append(("Description", m.description))
    if kind == "base" and m.creator_name:
        rows.append(("Creator", m.creator_name))
    if kind == "output" and m.training_run_name:
        rows.append(("Training Run", m.training_run_name))
    if m.created_at:
        rows.append(("Created", format_date(m.created_at)))
    url = platform_entity_url(ws_name, project["project_name"], "models", m.id)
    rows.append(("URL", url))

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

    url = platform_entity_url(ws_name, detail.project_name)
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
        rows.append(("Created", format_date(detail.created_at)))
    rows.append(("URL", url))

    console.table(rows, title="Project Info")
    console.print()
    return True


def _open_in_browser(ws_name: str, project: dict) -> None:
    """Open the current workspace/project URL in the default browser."""
    url = platform_entity_url(ws_name, project["project_name"])
    console.print(f"Opening {console.format_styled(url, 'dim')} ...")
    webbrowser.open(url)
    console.print()


@app.callback(invoke_without_command=True)
def workspace() -> None:
    """Manage workspace and project context."""
    workspaces = get_all_workspaces()

    if not workspaces:
        raise CLIError(MSG_NOT_LOGGED_IN)

    active_ws = next((name for name, _, is_active in workspaces if is_active), None)
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
        elif action in ("runs", "models", "datasets", "info", "browser"):
            if ws_name is None or default_project is None:
                continue

            if action == "runs":
                ok = _browse_runs(ws_name, default_project)
            elif action == "models":
                ok = _browse_models(ws_name, default_project)
            elif action == "datasets":
                ok = _browse_datasets(ws_name, default_project)
            elif action == "info":
                ok = _show_project_info(ws_name, default_project)
            else:
                _open_in_browser(ws_name, default_project)
                ok = True

            if not ok:
                default_project = None  # Already cleared by _handle_stale_project
                _show_context(ws_name, default_project)
