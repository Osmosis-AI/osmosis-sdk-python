"""Benchmark commands (submit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage benchmark runs (submit).",
    no_args_is_help=True,
)


@app.command("submit")
def benchmark_submit(
    config_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        resolve_path=False,
        help="Path to benchmark config TOML file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Submit a benchmark run."""
    from osmosis_ai.platform.cli.benchmark import submit as _submit

    return _submit(config_path, yes=yes)
