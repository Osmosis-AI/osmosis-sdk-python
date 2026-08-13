"""Dataset management commands (thin shell delegating to platform/cli/dataset.py)."""

from __future__ import annotations

import typer

from osmosis_ai.cli.options import (
    all_option,
    cursor_option,
    limit_option,
    log_limit_option,
)
from osmosis_ai.cli.output import CommandResult

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, download, list, info, preview, validate).",
    no_args_is_help=True,
)


@app.command("upload")
def upload(
    file: str = typer.Argument(..., help="Path to the file to upload."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing dataset with the same name.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> CommandResult:
    """Upload a dataset file."""
    from osmosis_ai.platform.cli.dataset import upload as _upload

    return _upload(file=file, overwrite=overwrite, yes=yes)


@app.command("download")
def download(
    name: str = typer.Argument(..., help="Dataset name."),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination file or existing directory.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the destination file if it already exists.",
    ),
) -> CommandResult:
    """Download a dataset file."""
    from osmosis_ai.platform.cli.dataset import download as _download

    return _download(name=name, output=output, overwrite=overwrite)


@app.command("list")
def list_datasets(
    limit: int = limit_option("Maximum number of datasets to show."),
    all_: bool = all_option("Show all datasets."),
) -> CommandResult:
    """List datasets."""
    from osmosis_ai.platform.cli.dataset import list_datasets as _list_datasets

    return _list_datasets(limit=limit, all_=all_)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Dataset name."),
) -> CommandResult:
    """Show dataset details and processing status."""
    from osmosis_ai.platform.cli.dataset import info as _info

    return _info(name=name)


@app.command("logs")
def logs(
    name: str = typer.Argument(..., help="Dataset name."),
    limit: int = log_limit_option(),
    cursor: str | None = cursor_option(),
) -> CommandResult:
    """Show recent logs for a dataset, oldest first."""
    from osmosis_ai.platform.cli.dataset import logs as _logs

    return _logs(name, limit=limit, cursor=cursor)


@app.command("preview")
def preview(
    name: str = typer.Argument(..., help="Dataset name."),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
) -> CommandResult:
    """Preview dataset rows."""
    from osmosis_ai.platform.cli.dataset import preview as _preview

    return _preview(name=name, rows=rows)


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> CommandResult:
    """Validate a dataset file locally."""
    from osmosis_ai.platform.cli.dataset import validate as _validate

    return _validate(file=file)
