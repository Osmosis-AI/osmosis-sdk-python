from __future__ import annotations

import argparse
import os

from osmosis_ai.cli.console import console
from osmosis_ai.platform.auth import (
    LoginError,
    delete_credentials,
    load_credentials,
    login,
)


class LoginCommand:
    """Handler for `osmosis login`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-f",
            "--force",
            dest="force",
            action="store_true",
            help="Force re-login, clearing existing credentials.",
        )
        parser.add_argument(
            "--no-browser",
            dest="no_browser",
            action="store_true",
            help="Don't open browser automatically, just print the URL.",
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

    def run(self, args: argparse.Namespace) -> int:
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80

        if term_width >= self.ASCII_ART_MIN_WIDTH:
            print(self.ASCII_ART)
        else:
            console.print()
            console.print("  Osmosis AI", style="bold magenta")
            console.print()

        try:
            # Clear existing credentials if forcing re-login
            if args.force and load_credentials():
                delete_credentials()

            result = login(no_browser=args.no_browser)

            info_lines = [f"Email: {result.user.email}"]
            if result.user.name:
                info_lines.append(f"Name: {result.user.name}")
            info_lines.append(
                f"Workspace: {result.organization.name} ({result.organization.role})"
            )
            info_lines.append(f"Expires: {result.expires_at.strftime('%Y-%m-%d')}")

            console.panel("Login Successful", "\n".join(info_lines), style="green")

            if result.revoked_previous_tokens > 0:
                token_word = (
                    "token" if result.revoked_previous_tokens == 1 else "tokens"
                )
                console.print(
                    f"[Note] {result.revoked_previous_tokens} previous {token_word} for this device was revoked",
                    style="dim",
                )

            console.print(
                "\nRun 'osmosis workspace' to select a default project.", style="dim"
            )

            return 0

        except LoginError as e:
            console.print_error(str(e))
            return 1
        except KeyboardInterrupt:
            console.print("\n\nLogin cancelled.")
            return 1
