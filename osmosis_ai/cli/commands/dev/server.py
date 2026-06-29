from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_LOG_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

app: typer.Typer = typer.Typer(
    help="Manage a remote rollout server.", no_args_is_help=True
)


@app.command("up")
def up(
    no_ttl: bool = typer.Option(
        False, "--no-ttl", help="Disable the 24h auto-teardown."
    ),
    ttl_hours: int = typer.Option(
        24, "--ttl-hours", min=1, help="Hours before auto-teardown."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Provision a remote rollout server for the current rollout folder."""
    from osmosis_ai.platform.cli.dev_server import up as _up

    return _up(ttl_hours=None if no_ttl else ttl_hours, yes=yes)


@app.command("down")
def down(
    server_id: str = typer.Argument(..., help="The rollout server id from `up`."),
) -> Any:
    """Tear down a remote rollout server."""
    from osmosis_ai.platform.cli.dev_server import down as _down

    return _down(server_id)


@app.command("logs")
def logs(
    server_id: str = typer.Argument(..., help="The rollout server id from `up`."),
    follow: bool = typer.Option(
        None,
        "-f",
        "--follow",
        help="Stream new logs as they arrive (default in rich mode).",
    ),
    tail: int = typer.Option(
        100,
        "--tail",
        "-n",
        min=1,
        max=MAX_LOG_PAGE_SIZE,
        help="Number of recent log lines to show.",
    ),
) -> Any:
    """Show logs for a remote rollout server."""
    from osmosis_ai.platform.cli.dev_server import logs as _logs

    return _logs(server_id, follow=follow, tail=tail)


@app.command("list")
def list_servers(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of rollout servers to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all rollout servers."),
) -> Any:
    """List active rollout servers for the current workspace."""
    from osmosis_ai.platform.cli.dev_server import list_servers as _list

    return _list(limit=limit, all_=all_)
