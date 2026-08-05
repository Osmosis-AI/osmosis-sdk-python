"""Benchmark catalog and run management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from osmosis_ai.platform.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_LOG_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

app: typer.Typer = typer.Typer(
    help="Manage benchmarks and their runs.",
    no_args_is_help=True,
)
runs_app: typer.Typer = typer.Typer(
    help="Manage benchmark runs.",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")


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
    key: str = typer.Argument(..., help="Benchmark key."),
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of runs to show in the runs section.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all of the benchmark's runs."),
) -> Any:
    """Show a benchmark: metadata, task options, leaderboard, and runs."""
    from osmosis_ai.platform.cli.benchmark import benchmark_info as _info

    return _info(key, limit=limit, all_=all_)


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


@runs_app.command("list")
def benchmark_runs_list(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_PAGE_SIZE,
        help="Maximum number of benchmark runs to show.",
    ),
    all_: bool = typer.Option(False, "--all", help="Show all benchmark runs."),
) -> Any:
    """List benchmark runs for the current workspace directory."""
    from osmosis_ai.platform.cli.benchmark import list_benchmark_runs as _list

    return _list(limit=limit, all_=all_)


@runs_app.command("info")
def benchmark_runs_info(
    name: str = typer.Argument(..., help="Benchmark run name."),
) -> Any:
    """Show benchmark run details, progress, and results."""
    from osmosis_ai.platform.cli.benchmark import run_info as _info

    return _info(name)


@runs_app.command("logs")
def benchmark_runs_logs(
    name: str = typer.Argument(..., help="Benchmark run name."),
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
    """Show recent logs for a benchmark run, oldest first."""
    from osmosis_ai.platform.cli.benchmark import logs as _logs

    return _logs(name, limit=limit, cursor=cursor)


@runs_app.command("stop")
def benchmark_runs_stop(
    name: str = typer.Argument(..., help="Benchmark run name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a benchmark run."""
    from osmosis_ai.platform.cli.benchmark import stop as _stop

    return _stop(name, yes=yes)


@runs_app.command("download")
def benchmark_runs_download(
    name: str = typer.Argument(..., help="Benchmark run name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Run output root (default: .osmosis/benchmarks/<run-name>/).",
    ),
    types: str = typer.Option(
        "summary,results",
        "--type",
        help=(
            "Comma-separated selector: summary, results, artifacts, logs, all. "
            "Replaces the default selection."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-download files that already exist locally.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip size confirmation.",
    ),
) -> Any:
    """Download benchmark run summary, results, artifacts, or logs."""
    from osmosis_ai.platform.cli.benchmark import download as _download

    return _download(
        name,
        output=output,
        types=types,
        overwrite=overwrite,
        yes=yes,
    )
