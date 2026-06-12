"""Training run management commands (thin shells delegating to platform/cli/train.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.platform.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_LOG_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


@app.command("list")
def list_runs(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of runs to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all training runs."),
) -> Any:
    """List training runs for the current workspace directory."""
    from osmosis_ai.platform.cli.train import list_training_runs as _list

    return _list(limit=limit, all_=all_)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Training run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output path for metrics JSON. Non-.json extensions are replaced with"
            " .json; a trailing '/' or existing directory generates a default"
            " filename inside it. (default in rich mode: .osmosis/metrics/)"
        ),
    ),
) -> Any:
    """Show training run details, checkpoints, and metrics."""
    from osmosis_ai.platform.cli.train import info as _info

    return _info(name, output=output)


@app.command("logs")
def logs(
    name: str = typer.Argument(..., help="Training run name."),
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_LOG_PAGE_SIZE,
        help="Maximum number of recent log entries to show.",
    ),
    cursor: str | None = typer.Option(
        None,
        "--cursor",
        help="Page further back using the next_cursor value from a previous page.",
    ),
) -> Any:
    """Show recent logs for a training run, oldest first."""
    from osmosis_ai.platform.cli.train import logs as _logs

    return _logs(name, limit=limit, cursor=cursor)


@app.command("submit")
def submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to training config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a new training run."""
    from osmosis_ai.platform.cli.train import submit as _submit

    return _submit(config_path, yes=yes)


@app.command("stop")
def stop(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a training run."""
    from osmosis_ai.platform.cli.train import stop as _stop

    return _stop(name, yes=yes)
