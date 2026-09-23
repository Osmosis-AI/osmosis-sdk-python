"""Rollout commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from osmosis_ai.cli.options import all_option, limit_option
from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Create, serve, and list rollouts.",
    no_args_is_help=True,
)


def _server_finished() -> NoReturn:
    """Exit cleanly after Uvicorn returns without invoking result rendering."""
    from osmosis_ai.cli.output import get_output_context

    get_output_context().output_emitted = True
    raise typer.Exit(0)


@app.command("serve")
def serve(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to rollout server config TOML file.",
        metavar="CONFIG",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Server bind host."),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Server port; defaults to _OSMOSIS_ROLLOUT_PORT or 8000.",
    ),
) -> NoReturn:
    """Run the rollout server declared by a TOML config."""
    from osmosis_ai.rollout.serve import serve as _serve

    _serve(config_path, host=host, port=port)
    _server_finished()


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
