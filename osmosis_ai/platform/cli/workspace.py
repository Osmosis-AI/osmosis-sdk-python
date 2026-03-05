from __future__ import annotations

import argparse
import webbrowser

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import Choice, Separator, confirm, is_interactive, select
from osmosis_ai.platform.api.models import (
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_SUCCESS,
)
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

from .utils import format_size

_BACK = "__back__"


class WorkspaceCommand:
    """Handler for `osmosis workspace`.

    Shows current context and provides an interactive navigation hub for
    workspace, project, and dataset management.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self._run)

    def _run(self, args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            raise CLIError("Not logged in. Run 'osmosis login' first.")

        active_ws = get_active_workspace()
        ws_name = active_ws
        default_project = get_default_project(active_ws) if active_ws else None

        # Validate the default project still exists
        default_project = self._validate_default_project(ws_name, default_project)

        # Show current context
        self._show_context(ws_name, default_project)

        # Non-interactive: just show current context
        if not is_interactive():
            return 0

        # Interactive main menu loop
        while True:
            action = self._main_menu(has_project=bool(default_project))

            if action is None or action == "exit":
                return 0
            elif action == "switch":
                result = self._switch_context(workspaces, active_ws)
                if result:
                    ws_name, default_project = result
                    active_ws = ws_name
                    console.print()
                    self._show_context(ws_name, default_project)
            elif action == "datasets":
                if not self._browse_datasets(ws_name, default_project):
                    default_project = None
                    self._show_context(ws_name, default_project)
            elif action == "info":
                if not self._show_project_info(ws_name, default_project):
                    default_project = None
                    self._show_context(ws_name, default_project)
            elif action == "browser":
                self._open_in_browser(ws_name, default_project)

    def _validate_default_project(
        self, ws_name: str | None, default_project: dict | None
    ) -> dict | None:
        """Check the default project still exists; clear it if stale."""
        if not ws_name or not default_project:
            return default_project

        from .project import _get_cached_projects

        project_id = default_project.get("project_id")
        if not project_id:
            return default_project

        projects = _get_cached_projects(max_age=0)
        if not projects:
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

    def _show_context(self, ws_name: str | None, default_project: dict | None) -> None:
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

    def _main_menu(self, has_project: bool) -> str | None:
        """Show main menu and return the selected action."""
        choices: list[Choice | Separator] = [
            Choice("Switch workspace / project", value="switch"),
        ]
        if has_project:
            choices.extend(
                [
                    Choice("Browse datasets", value="datasets"),
                    Choice("View project info", value="info"),
                    Choice("Open in browser", value="browser"),
                ]
            )
        choices.append(Separator())
        choices.append(Choice("Exit", value="exit"))

        console.separator()
        return select("Select an action", choices=choices)

    def _switch_context(
        self, workspaces: list[tuple], active_ws: str | None
    ) -> tuple[str, dict | None] | None:
        """Run the workspace → project → confirm flow.

        Each step has a Back option to return to the previous step.
        Back from workspace selection returns to the main menu.

        Returns (ws_name, project_dict) on success, or None if backed out to main menu.
        """
        step = "workspace"
        ws_name = None
        result = None
        while True:
            if step == "workspace":
                ws_name = self._select_workspace(workspaces, active_ws)
                if ws_name is None or ws_name == _BACK:
                    return None
                step = "project"

            elif step == "project":
                result = self._select_project(ws_name)
                if result == _BACK or result is None:
                    step = "workspace"
                    continue
                step = "confirm"

            elif step == "confirm":
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

    def _select_workspace(
        self,
        workspaces: list[tuple],
        active_ws: str | None,
    ) -> str | None:
        """Prompt the user to select a workspace. Returns _BACK or None to go back."""
        choices = []
        for name, creds, is_active in workspaces:
            marker = " (current)" if is_active else ""
            expired = " [expired]" if creds.is_expired() else ""
            title = f"{name}{marker}{expired}"
            choices.append(Choice(title, value=name))
        choices.append(Separator())
        choices.append(Choice("Back", value=_BACK))

        console.separator()
        return select("Select workspace:", choices=choices)

    def _select_project(self, ws_name: str) -> dict | str | None:
        """Prompt the user to select a default project.

        Returns:
            dict: Selected project with 'id' and 'project_name'.
            _BACK: User chose to go back to workspace selection.
            None: User cancelled or skipped.
        """
        from .project import select_project_interactive

        default = get_default_project(ws_name)
        current_id = default.get("project_id") if default else None

        result = select_project_interactive(
            ws_name, current_project_id=current_id, allow_back=True
        )

        if result == _BACK:
            return _BACK
        if result is None:
            return None
        return result

    def _browse_datasets(self, ws_name: str, project: dict) -> bool:
        """List datasets and allow selecting one for details.

        Returns False if the project was found to be stale (404).
        """
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
        project_id = project.get("project_id")
        try:
            result = client.list_datasets(project_id)
        except PlatformAPIError as e:
            if e.status_code == 404:
                clear_default_project(ws_name)
                console.print_error(
                    f"Project '{project.get('project_name', project_id)}' "
                    "no longer exists. Default project has been cleared."
                )
                return False
            else:
                console.print_error(f"Failed to load datasets: {e}")
            return True

        if not result.datasets:
            console.print("No datasets found.", style="dim")
            console.print()
            return True

        while True:
            choices: list[Choice | Separator] = []
            for d in result.datasets:
                status_str = self._format_dataset_status(d)
                label = f"{d.file_name} ({format_size(d.file_size)}) {status_str}"
                choices.append(Choice(label, value=d))
            choices.append(Separator())
            choices.append(Choice("Back", value=_BACK))

            console.separator()
            selected = select(
                f"Datasets ({result.total_count}):",
                choices=choices,
            )

            if selected is None or selected == _BACK:
                return True

            # Show dataset detail
            self._show_dataset_detail(selected, ws_name, project)

    def _show_dataset_detail(self, ds, ws_name: str, project: dict) -> None:
        """Display detailed info for a single dataset."""
        rows = [
            ("File", ds.file_name),
            ("ID", ds.id),
            ("Size", format_size(ds.file_size)),
            ("Status", ds.status),
        ]
        if ds.processing_step:
            pct = (
                f" ({ds.processing_percent:.0f}%)"
                if ds.processing_percent is not None
                else ""
            )
            rows.append(("Step", f"{ds.processing_step}{pct}"))
        if ds.error:
            rows.append(("Error", ds.error))
        if ds.created_at:
            rows.append(("Created", ds.created_at[:10]))

        url = (
            f"{PLATFORM_URL}/{ws_name}/{project['project_name']}/training-data/{ds.id}"
        )
        rows.append(("URL", url))

        console.table(rows, title="Dataset Detail")
        console.print()

    @staticmethod
    def _format_dataset_status(d) -> str:
        """Format a dataset status string with color styling."""
        if d.status in STATUSES_SUCCESS:
            color = "green"
        elif d.status in STATUSES_IN_PROGRESS:
            color = "yellow"
        elif d.status in STATUSES_ERROR:
            color = "red"
        else:
            color = None

        status_info = f"[{d.status}]"
        if d.processing_step:
            status_info = f"[{d.status}: {d.processing_step}]"

        if color:
            status_info = console.format_styled(status_info, color)
        return status_info

    def _show_project_info(self, ws_name: str, project: dict) -> bool:
        """Fetch and display project details.

        Returns False if the project was found to be stale (404).
        """
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
        project_id = project.get("project_id")
        try:
            detail = client.get_project(project_id)
        except PlatformAPIError as e:
            if e.status_code == 404:
                clear_default_project(ws_name)
                console.print_error(
                    f"Project '{project.get('project_name', project_id)}' "
                    "no longer exists. Default project has been cleared."
                )
                return False
            else:
                console.print_error(f"Failed to load project info: {e}")
            return True

        url = f"{PLATFORM_URL}/{ws_name}/{detail.project_name}"
        rows = [
            ("Project", detail.project_name),
            ("ID", detail.id),
            ("Role", detail.role),
            ("Datasets", str(detail.dataset_count)),
        ]
        if detail.created_at:
            rows.append(("Created", detail.created_at[:10]))
        rows.append(("URL", url))

        console.table(rows, title="Project Info")
        console.print()
        return True

    def _open_in_browser(self, ws_name: str, project: dict) -> None:
        """Open the current workspace/project URL in the default browser."""
        url = f"{PLATFORM_URL}/{ws_name}/{project['project_name']}"
        console.print(f"Opening {console.format_styled(url, 'dim')} ...")
        webbrowser.open(url)
        console.print()
