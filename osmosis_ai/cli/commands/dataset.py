"""Dataset management commands (thin shell delegating to platform/cli/dataset.py)."""

from __future__ import annotations

from typing import Any

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, download, list, info, preview, validate, delete).",
    no_args_is_help=True,
)


@app.command("upload")
def upload(
    file: str = typer.Argument(..., help="Path to the file to upload."),
) -> Any:
    """Upload a dataset file."""
    from osmosis_ai.platform.cli.dataset import upload as _upload

    return _upload(file=file)


@app.command("download")
def download(
    name: str = typer.Argument(..., help="Dataset name or ID."),
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
) -> Any:
    """Download a dataset file."""
    from osmosis_ai.platform.cli.dataset import download as _download

    return _download(name=name, output=output, overwrite=overwrite)


@app.command("list")
def list_datasets(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of datasets to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all datasets."),
) -> Any:
    """List datasets."""
    from osmosis_ai.platform.cli.dataset import list_datasets as _list_datasets

    return _list_datasets(limit=limit, all_=all_)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Dataset name."),
) -> Any:
    """Show dataset details and processing status."""
    from osmosis_ai.platform.cli.dataset import info as _info

    return _info(name=name)


@app.command("preview")
def preview(
    name: str = typer.Argument(..., help="Dataset name."),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
) -> Any:
    """Preview dataset rows."""
    from osmosis_ai.platform.cli.dataset import preview as _preview

    return _preview(name=name, rows=rows)


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> Any:
    """Validate a dataset file locally."""
    from osmosis_ai.platform.cli.dataset import validate as _validate

    return _validate(file=file)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Dataset name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> Any:
    """Delete a dataset."""
    from osmosis_ai.platform.cli.dataset import delete as _delete

    return _delete(name=name, yes=yes)
