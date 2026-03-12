from __future__ import annotations

import os

import typer

from osmosis_ai.cli.console import console
from osmosis_ai.platform.auth import (
    LoginError,
    delete_credentials,
    load_credentials,
    login,
)

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
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't open browser automatically, just print the URL.",
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

    try:
        # Clear existing credentials if forcing re-login
        if force and load_credentials():
            delete_credentials()

        result = login(no_browser=no_browser)

        esc = console.escape
        info_lines = [f"Email: {esc(result.user.email)}"]
        if result.user.name:
            info_lines.append(f"Name: {esc(result.user.name)}")
        info_lines.append(
            f"Workspace: {esc(result.organization.name)} ({esc(result.organization.role)})"
        )
        info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")

        console.panel("Login Successful", "\n".join(info_lines), style="green")

        if result.revoked_previous_tokens > 0:
            token_word = "token" if result.revoked_previous_tokens == 1 else "tokens"
            console.print(
                f"{esc('[Note]')} {result.revoked_previous_tokens} previous {token_word} for this device was revoked",
                style="dim",
            )

        console.print(
            "\nRun 'osmosis workspace' to select a default project.", style="dim"
        )

    except LoginError as e:
        console.print_error(str(e))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n\nLogin cancelled.")
        raise typer.Exit(1) from None
