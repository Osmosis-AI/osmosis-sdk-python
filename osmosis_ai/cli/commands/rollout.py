"""Rollout commands: list."""

from __future__ import annotations

from typing import Annotated

import typer

from osmosis_ai.cli.options import all_option, limit_option
from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Manage rollouts (init, list).",
    no_args_is_help=True,
)


@app.command("init")
def init(
    name: str = typer.Argument(
        ...,
        help="Rollout name (lowercase letters, digits, and hyphens).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help=(
            "Overwrite existing rollouts/<name>/ directory and configs/{eval,training}/"
            "<name>.toml. Without --force, the command refuses to clobber existing paths."
        ),
    ),
) -> CommandResult:
    """Scaffold a new rollout from the workspace template placeholders.

    Creates ``rollouts/<name>/{main.py,pyproject.toml,README.md}`` and
    ``configs/{eval,training}/<name>.toml`` so you can start editing right away.
    Must run inside an Osmosis workspace directory.
    """
    from osmosis_ai.templates.init import init_command

    return init_command(name=name, force=force)


@app.command("list")
def list_rollouts(
    limit: int = limit_option("Maximum number of rollouts to show."),
    all_: bool = all_option("Show all rollouts."),
    branch: Annotated[
        str | None,
        typer.Option("--branch", help="List rollouts synced from this branch."),
    ] = None,
) -> CommandResult:
    """List rollouts for the current workspace directory."""
    from osmosis_ai.platform.cli.rollout import list_rollouts as _list_rollouts

    return _list_rollouts(limit=limit, all_=all_, branch=branch)
