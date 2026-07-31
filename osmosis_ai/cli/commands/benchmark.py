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
    help="Manage benchmark runs.",
    no_args_is_help=True,
)
catalog_app: typer.Typer = typer.Typer(
    help="Discover available benchmarks and task-selection options.",
    no_args_is_help=True,
)
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("list")
def benchmark_catalog_list(
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


@catalog_app.command("info")
def benchmark_catalog_info(
    name_or_id: str = typer.Argument(..., help="Benchmark key, name, or ID."),
) -> Any:
    """Show benchmark metadata and task-selection options."""
    from osmosis_ai.platform.cli.benchmark import catalog_info as _info

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


@app.command("list")
def benchmark_list(
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


@app.command("info")
def benchmark_info(
    name_or_id: str = typer.Argument(..., help="Benchmark run name or ID."),
) -> Any:
    """Show benchmark run details, progress, and results."""
    from osmosis_ai.platform.cli.benchmark import run_info as _info

    return _info(name_or_id)


@app.command("logs")
def benchmark_logs(
    name_or_id: str = typer.Argument(..., help="Benchmark run name or ID."),
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

    return _logs(name_or_id, limit=limit, cursor=cursor)


@app.command("stop")
def benchmark_stop(
    name_or_id: str = typer.Argument(..., help="Benchmark run name or ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Stop a benchmark run."""
    from osmosis_ai.platform.cli.benchmark import stop as _stop

    return _stop(name_or_id, yes=yes)


@app.command("download")
def benchmark_download(
    name_or_id: str = typer.Argument(..., help="Benchmark run name or ID."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Run output root (default: .osmosis/benchmarks/<name>/).",
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
    """Download benchmark summary, results, artifacts, or logs."""
    from osmosis_ai.platform.cli.benchmark import download as _download

    return _download(
        name_or_id,
        output=output,
        types=types,
        overwrite=overwrite,
        yes=yes,
    )
