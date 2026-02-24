from __future__ import annotations

import argparse

from osmosis_ai.platform.auth import get_all_workspaces


class WhoamiCommand:
    """Handler for `osmosis whoami`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)

    def run(self, _args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            print("Not logged in. Run 'osmosis login' to authenticate.")
            return 1

        # Find active workspace for user info
        active_creds = None
        for _, creds, is_active in workspaces:
            if is_active:
                active_creds = creds
                break

        # Show user info from active workspace
        if active_creds:
            print(f"Email: {active_creds.user.email}")
            if active_creds.user.name:
                print(f"Name: {active_creds.user.name}")

        # Show all workspaces
        print(f"\nWorkspaces ({len(workspaces)}):")
        for name, creds, is_active in workspaces:
            active_marker = " *" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(
                f"  {name} ({creds.organization.role}){active_marker}{expired_marker}"
            )

        return 0
