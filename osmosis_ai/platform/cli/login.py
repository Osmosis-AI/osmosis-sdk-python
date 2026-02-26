from __future__ import annotations

import argparse

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

    def run(self, args: argparse.Namespace) -> int:
        ascii_art = """
                       ___           ___           ___           ___           ___                       ___
            ___       /\\  \\         /\\  \\         /\\__\\         /\\  \\         /\\  \\          ___        /\\  \\
      __   /\\__\\     /::\\  \\       /::\\  \\       /::|  |       /::\\  \\       /::\\  \\        /\\  \\      /::\\  \\
    /\\__\\  \\/__/    /:/\\:\\  \\     /:/\\ \\  \\     /:|:|  |      /:/\\:\\  \\     /:/\\ \\  \\       \\:\\  \\    /:/\\ \\  \\
   /:/  /  /\\__\\   /:/  \\:\\  \\   _\\:\\~\\ \\  \\   /:/|:|__|__   /:/  \\:\\  \\   _\\:\\~\\ \\  \\      /::\\__\\  _\\:\\~\\ \\  \\
  /:/  /  /:/  /  /:/__/ \\:\\__\\ /\\ \\:\\ \\ \\__\\ /:/ |::::\\__\\ /:/__/ \\:\\__\\ /\\ \\:\\ \\ \\__\\  __/:/\\/__/ /\\ \\:\\ \\ \\__\\
  \\/__/  /:/  /   \\:\\  \\ /:/  / \\:\\ \\:\\ \\/__/ \\/__/~~/:/  / \\:\\  \\ /:/  / \\:\\ \\:\\ \\/__/ /\\/:/  /    \\:\\ \\:\\ \\/__/
  /\\__\\  \\/__/     \\:\\  /:/  /   \\:\\ \\:\\__\\         /:/  /   \\:\\  /:/  /   \\:\\ \\:\\__\\   \\::/__/      \\:\\ \\:\\__\\
  \\/__/             \\:\\/:/  /     \\:\\/:/  /        /:/  /     \\:\\/:/  /     \\:\\/:/  /    \\:\\__\\       \\:\\/:/  /
                     \\::/  /       \\::/  /        /:/  /       \\::/  /       \\::/  /      \\/__/        \\::/  /
                      \\/__/         \\/__/         \\/__/         \\/__/         \\/__/                     \\/__/

"""
        print(ascii_art)

        try:
            # Clear existing credentials if forcing re-login
            if args.force and load_credentials():
                delete_credentials()

            result = login(no_browser=args.no_browser)

            print(f"\n[OK] Logged in as {result.user.email}")
            if result.user.name:
                print(f"    Name: {result.user.name}")
            print(
                f"    Workspace: {result.organization.name} ({result.organization.role})"
            )
            print(f"    Token expires: {result.expires_at.strftime('%Y-%m-%d')}")
            if result.revoked_previous_tokens > 0:
                token_word = (
                    "token" if result.revoked_previous_tokens == 1 else "tokens"
                )
                print(
                    f"    [Note] {result.revoked_previous_tokens} previous {token_word} for this device was revoked"
                )

            print("\nRun 'osmosis workspace' to select a default project.")

            return 0

        except LoginError as e:
            print(f"\n[ERROR] {e}")
            return 1
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            return 1
