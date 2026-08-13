"""Authentication commands: login, logout, whoami."""

from __future__ import annotations

import typer

from osmosis_ai.cli.output import (
    CommandResult,
    OperationResult,
    OutputFormat,
    get_output_context,
)

app: typer.Typer = typer.Typer(
    help="Manage authentication (login, logout, whoami).", no_args_is_help=True
)

ASCII_ART_MIN_WIDTH = 113
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


def _print_login_success(result: CommandResult) -> None:
    if (
        get_output_context().format is not OutputFormat.rich
        or not isinstance(result, OperationResult)
        or result.status != "success"
        or result.resource is None
    ):
        return

    from osmosis_ai.cli.console import console

    resource = result.resource
    info_lines = [f"Email: {console.escape(str(resource.get('email') or ''))}"]
    name = resource.get("name")
    if name:
        info_lines.append(f"Name: {console.escape(str(name))}")
    info_lines.append(
        f"Platform: {console.escape(str(resource.get('platform_url') or ''))}"
    )
    expires = str(resource.get("expires_at") or "").partition("T")[0]
    info_lines.append(f"Expires: {console.escape(expires)}")
    console.panel("Login Successful", "\n".join(info_lines), style="green")
    result.message = None


@app.command("login")
def login(
    force: bool = typer.Option(
        False, "-f", "--force", help="Force re-login, clearing existing credentials."
    ),
    token: str | None = typer.Option(
        None, "--token", help="Authenticate with a personal access token (for CI/CD)."
    ),
) -> CommandResult:
    """Authenticate with Osmosis AI."""
    from osmosis_ai.platform.cli.auth import login as _login

    if get_output_context().format is OutputFormat.rich:
        from osmosis_ai.cli.console import console

        if console.width >= ASCII_ART_MIN_WIDTH:
            console.print(ASCII_ART, markup=False, highlight=False)
        else:
            console.print()
            console.print("  Osmosis AI", style="bold magenta")
            console.print()
    result = _login(force=force, token=token)
    _print_login_success(result)
    return result


@app.command("logout")
def logout(
    skip_confirm: bool = typer.Option(
        False, "-y", "--yes", help="Skip confirmation prompt."
    ),
) -> CommandResult:
    """Logout from Osmosis AI CLI."""
    from osmosis_ai.platform.cli.auth import logout as _logout

    return _logout(skip_confirm=skip_confirm)


@app.command("whoami")
def whoami() -> CommandResult:
    """Show current authenticated user."""
    from osmosis_ai.platform.cli.auth import whoami as _whoami

    return _whoami()
