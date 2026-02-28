from __future__ import annotations

import argparse
from typing import Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.prompts import Choice, Separator, confirm, select
from osmosis_ai.platform.auth import (
    delete_workspace_credentials,
    get_all_workspaces,
)


class LogoutCommand:
    """Handler for `osmosis logout`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "--all",
            dest="logout_all",
            action="store_true",
            help="Logout from all workspaces.",
        )
        parser.add_argument(
            "-y",
            "--yes",
            dest="skip_confirm",
            action="store_true",
            help="Skip confirmation prompt.",
        )

    def run(self, args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            console.print("Not logged in.")
            return 0

        if args.logout_all:
            return self._logout_all(workspaces, args.skip_confirm)
        else:
            return self._logout_interactive(workspaces, args.skip_confirm)

    def _logout_all(
        self,
        workspaces: list[tuple[str, Any, bool]],
        skip_confirm: bool,
    ) -> int:
        """Logout from all workspaces."""
        workspace_names = [name for name, _, _ in workspaces]

        if not skip_confirm:
            console.print(f"This will logout from {len(workspaces)} workspace(s):")
            for name in workspace_names:
                console.print(f"  - {name}")

            result = confirm(
                f"Logout from all {len(workspaces)} workspace(s)?", default=False
            )
            if result is None:  # User cancelled with Ctrl+C
                return 0
            if not result:
                console.print("Cancelled.")
                return 0

        success_count = 0
        for name, _, _ in workspaces:
            if delete_workspace_credentials(name):
                success_count += 1

        console.print(
            f"Logged out from {success_count}/{len(workspaces)} workspace(s).",
            style="green",
        )
        return 0

    def _logout_interactive(
        self,
        workspaces: list[tuple[str, Any, bool]],
        skip_confirm: bool,
    ) -> int:
        """Interactive workspace selection for logout."""
        if len(workspaces) == 1:
            # Only one workspace, logout directly
            name, _, _ = workspaces[0]
            if not skip_confirm:
                result = confirm(f"Logout from '{name}'?", default=False)
                if result is None:  # User cancelled with Ctrl+C
                    return 0
                if not result:
                    console.print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                console.print(f"Logged out from '{name}'.", style="green")
            return 0

        # Multiple workspaces, show selection
        choices = []
        for name, creds, is_active in workspaces:
            active_marker = " (active)" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            title = f"{name}{active_marker}{expired_marker}"
            choices.append(Choice(title=title, value=name))

        choices.append(Separator())
        choices.append(Choice(title="All workspaces", value="__all__"))

        selected = select("Select workspace to logout from:", choices=choices)
        if selected is None:  # User cancelled with Ctrl+C
            return 0

        if selected == "__all__":
            return self._logout_all(workspaces, skip_confirm)
        else:
            # Selected a specific workspace
            name = selected
            if not skip_confirm:
                result = confirm(f"Logout from '{name}'?", default=False)
                if result is None:  # User cancelled with Ctrl+C
                    return 0
                if not result:
                    console.print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                console.print(f"Logged out from '{name}'.", style="green")
            return 0
