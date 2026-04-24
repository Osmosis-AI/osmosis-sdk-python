"""Dataset management commands (thin shell delegating to platform/cli/dataset.py)."""

from __future__ import annotations

import typer

from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, list, info, preview, validate, delete).",
    no_args_is_help=True,
)


@app.command("upload")
def upload(
    file: str = typer.Argument(..., help="Path to the file to upload."),
) -> None:
    """Upload a dataset file."""
    from osmosis_ai.platform.cli.dataset import upload as _upload

    _upload(file=file)


@app.command("list")
def list_datasets(
    limit: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--limit", help="Maximum number of datasets to show."
    ),
    all_: bool = typer.Option(False, "--all", help="Show all datasets."),
) -> None:
    """List datasets."""
    from osmosis_ai.platform.cli.dataset import list_datasets as _list_datasets

    _list_datasets(limit=limit, all_=all_)


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Dataset name."),
) -> None:
    """Show dataset details and processing status."""
    from osmosis_ai.platform.cli.dataset import info as _info

    _info(name=name)


@app.command("preview")
def preview(
    name: str = typer.Argument(..., help="Dataset name."),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
) -> None:
    """Preview dataset rows."""
    from osmosis_ai.platform.cli.dataset import preview as _preview

    _preview(name=name, rows=rows)


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> None:
    """Validate a dataset file locally."""
    from osmosis_ai.platform.cli.dataset import validate as _validate

    _validate(file=file)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Dataset name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a dataset."""
    from osmosis_ai.platform.cli.dataset import delete as _delete

    _delete(name=name, yes=yes)
