"""Authentication commands: login, logout, whoami."""

from __future__ import annotations

import typer

from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Manage authentication (login, logout, whoami).", no_args_is_help=True
)


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

    return _login(force=force, token=token)


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
