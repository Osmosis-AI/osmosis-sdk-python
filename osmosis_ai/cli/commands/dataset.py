"""Dataset management commands (thin shell delegating to platform/cli/dataset.py)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, list, status, preview, validate, delete).",
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
def list_datasets() -> None:
    """List datasets."""
    from osmosis_ai.platform.cli.dataset import list_datasets as _list_datasets

    _list_datasets()


@app.command("status")
def status(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
) -> None:
    """Check dataset processing status."""
    from osmosis_ai.platform.cli.dataset import status as _status

    _status(id=id)


@app.command("preview")
def preview(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
) -> None:
    """Preview dataset rows."""
    from osmosis_ai.platform.cli.dataset import preview as _preview

    _preview(id=id, rows=rows)


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> None:
    """Validate a dataset file locally."""
    from osmosis_ai.platform.cli.dataset import validate as _validate

    _validate(file=file)


@app.command("delete")
def delete(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete a dataset."""
    from osmosis_ai.platform.cli.dataset import delete as _delete

    _delete(id=id, yes=yes)
