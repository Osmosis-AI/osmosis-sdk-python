"""Workspace secret management commands.

``osmosis secret`` manages workspace environment secrets — the same records
referenced by the ``[secrets]`` table in train/eval configs. The platform
never returns secret values: ``list`` shows names + metadata only, and
``add`` echoes back only metadata.

Secret values are accepted from a hidden interactive prompt or from a named
environment variable (``--env VARNAME``) — never as a plaintext command-line
argument, which would leak into shell history and the process list.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage workspace secrets (list, add).",
    no_args_is_help=True,
)


@app.command("list")
def list_secrets(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of secrets to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all secrets."),
) -> Any:
    """List workspace secrets (names and metadata only; values are never shown)."""
    from osmosis_ai.platform.cli.secret import list_secrets as _list_secrets

    return _list_secrets(limit=limit, all_=all_)


@app.command("add")
def add_secret(
    name: str = typer.Argument(..., help="Secret name.", metavar="NAME"),
    env: str | None = typer.Option(
        None,
        "--env",
        metavar="VARNAME",
        help=(
            "Read the secret value from the named environment variable "
            "(e.g. --env OPENAI_API_KEY). Without this flag you are prompted "
            "to type the value interactively (input is hidden)."
        ),
    ),
) -> Any:
    """Add a workspace secret.

    The value is read from --env VARNAME, or typed at a hidden interactive
    prompt. It is never accepted as a plaintext command-line argument.
    """
    from osmosis_ai.platform.cli.secret import add_secret as _add_secret

    return _add_secret(name=name, env=env)
