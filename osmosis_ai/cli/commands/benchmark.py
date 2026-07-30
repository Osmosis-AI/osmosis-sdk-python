"""Benchmark catalog and run submission commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Discover benchmarks and submit benchmark runs (list, info, submit).",
    no_args_is_help=True,
)


@app.command("list")
def benchmark_list(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of benchmarks to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all benchmarks."),
) -> Any:
    """List benchmarks available in the current workspace."""
    from osmosis_ai.platform.cli.benchmark import list_benchmarks as _list_benchmarks

    return _list_benchmarks(limit=limit, all_=all_)


@app.command("info")
def benchmark_info(
    name_or_id: str = typer.Argument(..., help="Benchmark name or ID."),
) -> Any:
    """Show benchmark metadata and task-selection options."""
    from osmosis_ai.platform.cli.benchmark import info as _info

    return _info(name_or_id)


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
