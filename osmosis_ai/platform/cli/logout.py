from __future__ import annotations

import argparse
from typing import Any

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
            print("Not logged in.")
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
            print(f"This will logout from {len(workspaces)} workspace(s):")
            for name in workspace_names:
                print(f"  - {name}")
            confirm = input("\nAre you sure? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Cancelled.")
                return 0

        success_count = 0
        for name, _, _ in workspaces:
            if delete_workspace_credentials(name):
                success_count += 1

        print(f"Logged out from {success_count}/{len(workspaces)} workspace(s).")
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
                confirm = input(f"Logout from '{name}'? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                print(f"Logged out from '{name}'.")
            return 0

        # Multiple workspaces, show selection
        print("Select workspace to logout from:\n")
        for i, (name, creds, is_active) in enumerate(workspaces, 1):
            active_marker = " (active)" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(f"  {i}. {name}{active_marker}{expired_marker}")
        print(f"  {len(workspaces) + 1}. All workspaces")
        print("  0. Cancel")

        try:
            choice = input("\nEnter number: ").strip()
            choice_num = int(choice)
        except (ValueError, EOFError):
            print("Cancelled.")
            return 0

        if choice_num == 0:
            print("Cancelled.")
            return 0
        elif choice_num == len(workspaces) + 1:
            return self._logout_all(workspaces, skip_confirm)
        elif 1 <= choice_num <= len(workspaces):
            name, _, _ = workspaces[choice_num - 1]
            if not skip_confirm:
                confirm = input(f"Logout from '{name}'? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                print(f"Logged out from '{name}'.")
            return 0
        else:
            print("Invalid selection.")
            return 1
