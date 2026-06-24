from __future__ import annotations

from typing import Any

import typer

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
) -> Any:
    """Provision a remote rollout server for the current rollout folder."""
    from osmosis_ai.platform.cli.dev_server import up as _up

    return _up(ttl_hours=None if no_ttl else ttl_hours)


@app.command("down")
def down(
    server_id: str = typer.Argument(..., help="The rollout server id from `up`."),
) -> Any:
    """Tear down a remote rollout server."""
    from osmosis_ai.platform.cli.dev_server import down as _down

    return _down(server_id)
