from __future__ import annotations

import os

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.cli.prompts import confirm
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.platform.auth.local_config import reset_session
from osmosis_ai.platform.auth.platform_client import revoke_cli_token


def logout(
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> None:
    """Logout from Osmosis AI CLI."""
    credentials = load_credentials()

    if credentials is None:
        console.print("Not logged in.")
        return

    if not skip_confirm:
        result = confirm("Logout from Osmosis AI?", default=False)
        if result is None:  # User cancelled with Ctrl+C
            return
        if not result:
            console.print("Cancelled.")
            return

    # Best-effort server-side revocation
    if not credentials.is_expired():
        revoke_cli_token(credentials)

    # Delete local credentials and workspace/project state
    reset_session()

    console.print("Logged out successfully.", style="green")

    if os.environ.get("OSMOSIS_TOKEN"):
        console.print(
            "Warning: OSMOSIS_TOKEN environment variable is still set. "
            "Run 'unset OSMOSIS_TOKEN' to fully logout.",
            style="yellow",
        )
