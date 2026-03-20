from __future__ import annotations

import os

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.auth import (
    LoginError,
    device_login,
    load_credentials,
    verify_token,
)
from osmosis_ai.platform.auth.credentials import (
    Credentials,
    delete_credentials,
    save_credentials,
)
from osmosis_ai.platform.auth.flow import LoginResult
from osmosis_ai.platform.auth.local_config import (
    clear_all_local_data,
    get_active_workspace,
    set_active_workspace,
)
from osmosis_ai.platform.auth.platform_client import platform_request, revoke_cli_token


def _validate_workspace_context(creds: Credentials) -> None:
    """Validate the stored workspace is still accessible with new credentials.

    After login, the workspace ID in config.json may be stale (e.g. the user
    switched between local dev and production, or the local DB was recreated).
    This check prevents confusing 403 errors on subsequent commands.
    """
    ws = get_active_workspace()
    if not ws:
        return

    try:
        data = platform_request(
            "/api/cli/workspaces",
            credentials=creds,
            require_workspace=False,
            cleanup_on_401=False,
        )
        workspaces = data.get("workspaces", [])
        ws_by_id = {w["id"]: w for w in workspaces if "id" in w}
        ws_by_name = {w["name"]: w for w in workspaces if "name" in w}

        if ws["id"] in ws_by_id:
            return  # Still valid

        # ID is stale — try to fix by matching workspace name
        if ws["name"] in ws_by_name:
            correct = ws_by_name[ws["name"]]
            set_active_workspace(correct["id"], correct["name"])
            return

        # Workspace no longer accessible at all
        clear_all_local_data()
        console.print(
            "\nPrevious workspace is no longer accessible. "
            "Run 'osmosis workspace' to select a workspace.",
            style="yellow",
        )
    except Exception:
        pass  # Don't block login for validation errors


ASCII_ART = r"""
                       ___           ___           ___           ___           ___                       ___
            ___       /\  \         /\  \         /\__\         /\  \         /\  \          ___        /\  \
      __   /\__\     /::\  \       /::\  \       /::|  |       /::\  \       /::\  \        /\  \      /::\  \
    /\__\  \/__/    /:/\:\  \     /:/\ \  \     /:|:|  |      /:/\:\  \     /:/\ \  \       \:\  \    /:/\ \  \
   /:/  /  /\__\   /:/  \:\  \   _\:\~\ \  \   /:/|:|__|__   /:/  \:\  \   _\:\~\ \  \      /::\__\  _\:\~\ \  \
  /:/  /  /:/  /  /:/__/ \:\__\ /\ \:\ \ \__\ /:/ |::::\__\ /:/__/ \:\__\ /\ \:\ \ \__\  __/:/\/__/ /\ \:\ \ \__\
  \/__/  /:/  /   \:\  \ /:/  / \:\ \:\ \/__/ \/__/~~/:/  / \:\  \ /:/  / \:\ \:\ \/__/ /\/:/  /    \:\ \:\ \/__/
  /\__\  \/__/     \:\  /:/  /   \:\ \:\__\         /:/  /   \:\  /:/  /   \:\ \:\__\   \::/__/      \:\ \:\__\
  \/__/             \:\/:/  /     \:\/:/  /        /:/  /     \:\/:/  /     \:\/:/  /    \:\__\       \:\/:/  /
                     \::/  /       \::/  /        /:/  /       \::/  /       \::/  /      \/__/        \::/  /
                      \/__/         \/__/         \/__/         \/__/         \/__/                     \/__/
"""
ASCII_ART_MIN_WIDTH = 113


def login_cmd(
    force: bool = typer.Option(
        False, "-f", "--force", help="Force re-login, clearing existing credentials."
    ),
    token: str | None = typer.Option(
        None, "--token", help="Authenticate with a personal access token (for CI/CD)."
    ),
) -> None:
    """Authenticate with Osmosis AI."""
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80

    if term_width >= ASCII_ART_MIN_WIDTH:
        print(ASCII_ART)
    else:
        console.print()
        console.print("  Osmosis AI", style="bold magenta")
        console.print()

    # Refuse to login if OSMOSIS_TOKEN env var is set
    if os.environ.get("OSMOSIS_TOKEN"):
        console.print_error(
            "The OSMOSIS_TOKEN environment variable is set. "
            "Unset it to use 'osmosis login'."
        )
        raise typer.Exit(1)

    try:
        old_credentials = load_credentials()

        # Clear existing credentials if forcing re-login
        if force and old_credentials:
            delete_credentials()

        # Two login paths: token or device flow
        if token:
            verified = verify_token(token)
            creds = Credentials.from_verify_result(token, verified)
            result = LoginResult.from_verify_result(verified)
        else:
            result, creds = device_login()

        # Revoke old token server-side before overwriting local credentials,
        # so it doesn't become an unmanageable orphan on the platform.
        if (
            old_credentials
            and not old_credentials.is_expired()
            and old_credentials.token_id
        ):
            revoke_cli_token(old_credentials)

        save_credentials(creds)

        # Clear stale workspace/project context when user identity changes
        # or when explicitly forcing a fresh start, to prevent subsequent
        # commands from sending the old workspace ID in X-Osmosis-Org.
        local_data_cleared = force or (
            old_credentials and old_credentials.user.id != creds.user.id
        )
        if local_data_cleared:
            clear_all_local_data()
        else:
            _validate_workspace_context(creds)

        # Display login success
        esc = console.escape

        info_lines = [f"Email: {esc(result.user.email)}"]
        if result.user.name:
            info_lines.append(f"Name: {esc(result.user.name)}")
        info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")

        console.panel("Login Successful", "\n".join(info_lines), style="green")

        console.print(
            "\nRun 'osmosis workspace' to select a workspace.",
            style="dim",
        )

    except LoginError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n\nLogin cancelled.")
        raise typer.Exit(1) from None
