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
from osmosis_ai.platform.auth.local_config import clear_all_local_data

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

        store = save_credentials(creds)

        # Clear stale workspace/project context when user identity changes
        # or when explicitly forcing a fresh start, to prevent subsequent
        # commands from sending the old workspace ID in X-Osmosis-Org.
        if force or (old_credentials and old_credentials.user.id != creds.user.id):
            clear_all_local_data()

        # Display login success
        esc = console.escape

        info_lines = [f"Email: {esc(result.user.email)}"]
        if result.user.name:
            info_lines.append(f"Name: {esc(result.user.name)}")
        info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")
        info_lines.append(f"Token stored in: {store}")

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
