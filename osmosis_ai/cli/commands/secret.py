"""Secret management commands.

``osmosis secret`` manages secrets — the same records referenced by the
``[secrets]`` table in train/eval configs. Secrets are scoped: a ``workspace``
secret is shared across the workspace (admin/owner only), and a ``personal``
secret is private to the calling user. When both exist with the same name, the
personal secret takes precedence at run time.

The platform never returns secret values: ``list`` shows names + metadata
only, and ``set`` echoes back only metadata.

Secret values are accepted from a hidden interactive prompt or from a named
environment variable (``--env VARNAME``) — never as a plaintext command-line
argument, which would leak into shell history and the process list.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage secrets (set, list, delete).",
    no_args_is_help=True,
)

_SCOPE_HELP = (
    "Secret scope: 'workspace' (shared across the workspace, applies to "
    "everyone's runs by default) or 'personal' (overrides "
    "workspace secrets and applies only to runs you submit)."
)


@app.command("list")
def list_secrets(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        help="Maximum number of secrets to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all secrets."),
    scope: str = typer.Option(
        "all",
        "--scope",
        help="Filter by scope: 'all', 'workspace', or 'personal'.",
    ),
) -> Any:
    """List secrets (names and metadata only; values are never shown)."""
    from osmosis_ai.platform.cli.secret import list_secrets as _list_secrets

    return _list_secrets(limit=limit, all_=all_, scope=scope)


@app.command("set")
def set_secret(
    name: str = typer.Argument(..., help="Secret name.", metavar="NAME"),
    scope: str = typer.Option(..., "--scope", help=_SCOPE_HELP),
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
    """Create or update a secret (upsert).

    The value is read from --env VARNAME, or typed at a hidden interactive
    prompt. It is never accepted as a plaintext command-line argument.
    """
    from osmosis_ai.platform.cli.secret import set_secret as _set_secret

    return _set_secret(name=name, scope=scope, env=env)


@app.command("delete")
def delete_secret(
    name: str = typer.Argument(..., help="Secret name.", metavar="NAME"),
    scope: str = typer.Option(..., "--scope", help=_SCOPE_HELP),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> Any:
    """Delete a secret within the given scope."""
    from osmosis_ai.platform.cli.secret import delete_secret as _delete_secret

    return _delete_secret(name=name, scope=scope, yes=yes)
