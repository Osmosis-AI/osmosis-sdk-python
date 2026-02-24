from __future__ import annotations

import argparse

from osmosis_ai.platform.auth import (
    get_active_workspace,
    get_all_workspaces,
    set_active_workspace,
)


class WorkspaceCommand:
    """Handler for `osmosis workspace`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest="workspace_action", help="Workspace management commands"
        )

        # workspace list
        list_parser = subparsers.add_parser(
            "list", help="List all logged-in workspaces"
        )
        list_parser.set_defaults(handler=self._run_list)

        # workspace switch
        switch_parser = subparsers.add_parser(
            "switch", help="Switch to a different workspace"
        )
        switch_parser.add_argument("name", help="Name of the workspace to switch to")
        switch_parser.set_defaults(handler=self._run_switch)

        # workspace current
        current_parser = subparsers.add_parser(
            "current", help="Show the current active workspace"
        )
        current_parser.set_defaults(handler=self._run_current)

        # Default handler when no subcommand is provided
        parser.set_defaults(handler=self._run_default)

    def _run_default(self, args: argparse.Namespace) -> int:
        """Show help when no subcommand is provided."""
        print("Usage: osmosis workspace <command>")
        print("")
        print("Commands:")
        print("  list     List all logged-in workspaces")
        print("  switch   Switch to a different workspace")
        print("  current  Show the current active workspace")
        return 0

    def _run_list(self, args: argparse.Namespace) -> int:
        """List all stored workspaces."""
        workspaces = get_all_workspaces()

        if not workspaces:
            print(
                "No workspaces logged in. Run 'osmosis login' to log in to a workspace."
            )
            return 0

        print("Logged-in workspaces:")
        for name, creds, is_active in workspaces:
            active_marker = " (active)" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(
                f"  {name} ({creds.organization.role}){active_marker}{expired_marker}"
            )

        return 0

    def _run_switch(self, args: argparse.Namespace) -> int:
        """Switch to a different workspace."""
        workspace_name = args.name

        if set_active_workspace(workspace_name):
            print(f"Switched to workspace: {workspace_name}")
            return 0
        else:
            print(f"Workspace '{workspace_name}' not found.")
            print("Run 'osmosis workspace list' to see available workspaces.")
            return 1

    def _run_current(self, args: argparse.Namespace) -> int:
        """Show the current active workspace."""
        active = get_active_workspace()

        if not active:
            print("No active workspace. Run 'osmosis login' to log in to a workspace.")
            return 0

        workspaces = get_all_workspaces()
        for name, creds, is_active in workspaces:
            if is_active:
                expired_marker = " [expired]" if creds.is_expired() else ""
                print(
                    f"Current workspace: {name} ({creds.organization.role}){expired_marker}"
                )
                print(f"  User: {creds.user.email}")
                print(f"  Expires: {creds.expires_at.strftime('%Y-%m-%d')}")
                return 0

        print(f"Current workspace: {active}")
        return 0
