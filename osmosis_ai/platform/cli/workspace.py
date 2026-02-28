from __future__ import annotations

import argparse

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.prompts import Choice, is_interactive, select
from osmosis_ai.platform.auth import (
    get_active_workspace,
    get_all_workspaces,
    set_active_workspace,
)
from osmosis_ai.platform.auth.local_config import (
    get_default_project,
    set_default_project,
)


class WorkspaceCommand:
    """Handler for `osmosis workspace`.

    Shows current context and provides interactive workspace + project selection.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self._run)

    def _run(self, args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            raise CLIError("Not logged in. Run 'osmosis login' first.")

        active_ws = get_active_workspace()

        # Show current context
        if active_ws:
            default_project = get_default_project(active_ws)
            project_name = (
                default_project["project_name"]
                if default_project
                else console.format_styled("(no project selected)", "dim")
            )
            console.print(
                f"{console.format_styled('Current:', 'bold')} {console.format_styled(active_ws, 'cyan')} / {project_name}"
            )
            console.print()

        # Non-interactive: just show current context
        if not is_interactive():
            return 0

        # --- Workspace selection ---
        ws_name = self._select_workspace(workspaces, active_ws)
        if ws_name is None:
            raise CLIError("Cancelled.")

        switched = ws_name != active_ws
        if switched:
            set_active_workspace(ws_name)
            console.print(
                f"{console.format_styled('Switched to:', 'bold')} {console.format_styled(ws_name, 'cyan')}"
            )

        # --- Project selection (only if same workspace) ---
        if not switched:
            return self._select_project(ws_name)

        return 0

    def _select_workspace(
        self,
        workspaces: list[tuple],
        active_ws: str | None,
    ) -> str | None:
        """Prompt the user to select a workspace."""
        choices = []
        for name, creds, is_active in workspaces:
            marker = " (current)" if is_active else ""
            expired = " [expired]" if creds.is_expired() else ""
            title = f"{name}{marker}{expired}"
            choices.append(Choice(title, value=name))

        return select("Select workspace:", choices=choices)

    def _select_project(self, ws_name: str) -> int:
        """Prompt the user to select a default project."""
        from .project import select_project_interactive

        default = get_default_project(ws_name)
        current_id = default.get("project_id") if default else None

        result = select_project_interactive(ws_name, current_project_id=current_id)
        if result is None:
            return 0

        set_default_project(ws_name, result["id"], result["project_name"])
        console.print(
            f"{console.format_styled('Switched to:', 'bold')} {console.format_styled(ws_name, 'cyan')} / {console.format_styled(result['project_name'], 'cyan')}"
        )
        return 0
