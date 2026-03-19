from __future__ import annotations

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.platform.auth.credentials import get_credential_store
from osmosis_ai.platform.auth.local_config import get_active_workspace_name
from osmosis_ai.platform.cli.constants import MSG_NOT_LOGGED_IN

_STORE_LABELS = {
    "env": "environment variable",
    "keyring": "keyring",
    "file": "plain text",
}


def whoami() -> None:
    """Show current authenticated user and workspace."""
    credentials = load_credentials()

    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)

    store = get_credential_store()
    store_label = _STORE_LABELS.get(store or "", store or "unknown")
    ws_name = get_active_workspace_name()

    rows = [("Email", credentials.user.email)]
    if credentials.user.name:
        rows.append(("Name", credentials.user.name))
    if ws_name:
        rows.append(("Workspace", ws_name))
    rows.append(("Expires", credentials.expires_at.strftime("%Y-%m-%d")))
    rows.append(("Token", f"({store_label})"))

    console.table(rows)
