from __future__ import annotations

from typing import NoReturn

import typer

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.options import all_option, limit_option
from osmosis_ai.cli.output import CommandResult
from osmosis_ai.platform.constants import (
    MAX_LOG_PAGE_SIZE,
    DevServerBackend,
    DevServerSandboxEnvironment,
)

app: typer.Typer = typer.Typer(
    help="Manage a remote rollout server.", no_args_is_help=True
)


@app.command("up")
def up(
    url: str | None = typer.Option(
        None, "--url", help="GitHub Harbor task source; use the built-in gateway."
    ),
    path: str = typer.Option(
        "tasks",
        "--path",
        help="Task directory or dataset.toml relative to the repository root.",
    ),
    ref: str | None = typer.Option(
        None, "--ref", help="Full commit SHA from images build."
    ),
    no_ttl: bool = typer.Option(
        False, "--no-ttl", help="Disable the 24h auto-teardown."
    ),
    ttl_hours: int = typer.Option(
        24, "--ttl-hours", min=1, help="Hours before auto-teardown."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    backend: DevServerBackend | None = typer.Option(
        None,
        "--backend",
        help="Deployment backend for the rollout server (ecs or gke). Defaults to the platform setting.",
    ),
    sandbox_environment: DevServerSandboxEnvironment | None = typer.Option(
        None,
        "--sandbox-environment",
        help="Managed sandbox credentials for this server; defaults to the platform setting. The rollout code selects the matching backend.",
    ),
) -> CommandResult:
    """Provision a remote rollout server for the current rollout folder."""
    if url is None and (ref is not None or path != "tasks"):
        raise CLIError(
            "Source --path and --ref options require --url", code="VALIDATION"
        )
    from osmosis_ai.platform.cli.dev_server import up as _up

    return _up(
        ttl_hours=None if no_ttl else ttl_hours,
        yes=yes,
        sandbox_environment=sandbox_environment,
        **({"backend": backend} if backend is not None else {}),
        **({"url": url, "path": path, "ref": ref} if url is not None else {}),
    )


@app.command("down")
def down(
    url: str | None = typer.Option(
        None, "--url", help="Connected repository; no local checkout required."
    ),
    server_id: str = typer.Argument(..., help="The rollout server id from `up`."),
) -> CommandResult:
    """Tear down a remote rollout server."""
    from osmosis_ai.platform.cli.dev_server import down as _down

    return _down(server_id, **({"url": url} if url else {}))


@app.command("logs")
def logs(
    url: str | None = typer.Option(
        None, "--url", help="Connected repository; no local checkout required."
    ),
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
) -> NoReturn:
    """Show logs for a remote rollout server."""
    from osmosis_ai.platform.cli.dev_server import logs as _logs

    # _logs always raises typer.Exit or KeyboardInterrupt; there is no CommandResult.
    _logs(server_id, follow=follow, tail=tail, **({"url": url} if url else {}))


@app.command("list")
def list_servers(
    url: str | None = typer.Option(
        None, "--url", help="Connected repository; no local checkout required."
    ),
    limit: int = limit_option("Maximum number of rollout servers to show."),
    all_: bool = all_option("Show all rollout servers."),
) -> CommandResult:
    """List active rollout servers for the current workspace."""
    from osmosis_ai.platform.cli.dev_server import list_servers as _list

    return _list(limit=limit, all_=all_, **({"url": url} if url else {}))
