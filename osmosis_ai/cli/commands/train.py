"""Training run management commands (thin shells delegating to platform/cli/train.py)."""

from __future__ import annotations

from pathlib import Path

import typer

from osmosis_ai.cli.options import (
    all_option,
    cursor_option,
    limit_option,
    log_limit_option,
)
from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(help="Manage training runs.", no_args_is_help=True)


@app.command("list")
def list_runs(
    limit: int = limit_option("Maximum number of runs to show."),
    all_: bool = all_option("Show all training runs."),
) -> CommandResult:
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
) -> CommandResult:
    """Show training run details, checkpoints, and metrics."""
    from osmosis_ai.platform.cli.train import info as _info

    return _info(name, output=output)


@app.command("logs")
def logs(
    name: str = typer.Argument(..., help="Training run name."),
    limit: int = log_limit_option(),
    cursor: str | None = cursor_option(),
) -> CommandResult:
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
    secrets_file: str = typer.Option(
        None,
        "--secrets-file",
        help=(
            "Dotenv file supplying values for \\[secrets] names; - reads stdin. "
            "Values are never saved and are re-supplied on every run."
        ),
    ),
) -> CommandResult:
    """Submit a new training run."""
    from osmosis_ai.platform.cli.train import submit as _submit

    return _submit(config_path, yes=yes, secrets_file=secrets_file)


@app.command("stop")
def stop(
    name: str = typer.Argument(..., help="Training run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> CommandResult:
    """Stop a training run."""
    from osmosis_ai.platform.cli.train import stop as _stop

    return _stop(name, yes=yes)
