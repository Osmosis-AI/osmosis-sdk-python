"""Secret management commands.

``osmosis secret`` manages secrets — the same records referenced by the
``[secrets]`` section in submit configs. Evaluation configs must include
this section; default OpenAI eval configs should include ``OPENAI_API_KEY`` and
use ``required = []`` only when no secret refs are needed. Training configs may
omit ``[secrets]`` entirely, but any ``[secrets]`` section must include
``required``. Secrets are scoped: a ``workspace`` secret is shared across the
workspace (admin/owner only), and a ``personal`` secret is private to the
calling user. When both exist with the same name, the personal secret takes
precedence at run time.

The platform never returns secret values: ``list`` shows names + metadata
only, and ``set`` echoes back only metadata.

Secret values are accepted from a hidden interactive prompt or from a named
environment variable (``--env VARNAME``) — never as a plaintext command-line
argument, which would leak into shell history and the process list.
"""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage secrets (set, list, delete).",
    no_args_is_help=True,
)

_SCOPE_HELP = (
    "'personal' (default) or 'workspace'. Personal secrets apply only to "
    "your runs and override workspace secrets with the same name. "
    "Workspace secrets are shared across the workspace."
)


@app.command("list")
def list_secrets(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
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
    scope: str = typer.Option("personal", "--scope", help=_SCOPE_HELP),
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
    scope: str = typer.Option("personal", "--scope", help=_SCOPE_HELP),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> Any:
    """Delete a secret within the given scope."""
    from osmosis_ai.platform.cli.secret import delete_secret as _delete_secret

    return _delete_secret(name=name, scope=scope, yes=yes)
