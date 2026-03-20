from __future__ import annotations

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.platform.auth.local_config import get_active_workspace_name
from osmosis_ai.platform.cli.constants import MSG_NOT_LOGGED_IN


def whoami() -> None:
    """Show current authenticated user and workspace."""
    credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)

    ws_name = get_active_workspace_name()
    esc = console.escape

    rows = [("Email", esc(credentials.user.email))]
    if credentials.user.name:
        rows.append(("Name", esc(credentials.user.name)))
    if ws_name:
        rows.append(("Workspace", esc(ws_name)))
    rows.append(("Expires", credentials.expires_at.strftime("%Y-%m-%d")))

    console.table(rows)
