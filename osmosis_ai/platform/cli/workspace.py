from __future__ import annotations

import argparse
import sys

from osmosis_ai.cli.errors import CLIError
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
                else "(no project selected)"
            )
            print(f"Current: {active_ws} / {project_name}")
            print()

        # Non-interactive: just show current context
        if not sys.stdin.isatty():
            return 0

        # --- Workspace selection ---
        ws_name = self._select_workspace(workspaces, active_ws)

        if ws_name != active_ws:
            set_active_workspace(ws_name)

        # --- Project selection ---
        return self._select_project(ws_name)

    def _select_workspace(
        self,
        workspaces: list[tuple],
        active_ws: str | None,
    ) -> str:
        """Prompt the user to select a workspace. Auto-selects if only one."""
        if len(workspaces) == 1:
            return workspaces[0][0]

        print("Available workspaces:")
        current_idx = 1
        for i, (name, creds, is_active) in enumerate(workspaces, 1):
            marker = " (current)" if is_active else ""
            expired = " [expired]" if creds.is_expired() else ""
            print(f"  [{i}] {name}{marker}{expired}")
            if is_active:
                current_idx = i
        print()

        choice = _prompt(f"Select workspace [{current_idx}]: ")
        if choice is None:
            raise CLIError("Cancelled.")
        idx = current_idx - 1 if not choice else _parse_choice(choice, len(workspaces))

        return workspaces[idx][0]

    def _select_project(self, ws_name: str) -> int:
        """Prompt the user to select a default project."""
        from .project import select_project_interactive

        default = get_default_project(ws_name)
        current_id = default.get("project_id") if default else None

        result = select_project_interactive(ws_name, current_project_id=current_id)
        if result is None:
            return 0

        set_default_project(ws_name, result["id"], result["project_name"])
        print(f"Switched to: {ws_name} / {result['project_name']}")
        return 0


def _prompt(message: str) -> str | None:
    """Read input, returning None on interrupt/EOF."""
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _parse_choice(value: str, max_val: int) -> int:
    """Parse a 1-based numeric choice, returning a 0-based index."""
    try:
        idx = int(value) - 1
    except ValueError:
        raise CLIError("Invalid input.") from None
    if not (0 <= idx < max_val):
        raise CLIError("Invalid selection.")
    return idx
