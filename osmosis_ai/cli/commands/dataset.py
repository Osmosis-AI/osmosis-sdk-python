"""Dataset management commands (thin shell delegating to platform/cli/dataset.py)."""

from __future__ import annotations

import typer

from osmosis_ai.cli.errors import not_implemented

app: typer.Typer = typer.Typer(
    help="Manage datasets (upload, list, status, preview, validate, delete).",
    no_args_is_help=True,
)


@app.command("upload")
def upload(
    file: str = typer.Argument(..., help="Path to the file to upload."),
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
) -> None:
    """Upload a dataset file."""
    from osmosis_ai.platform.cli.dataset import upload as _upload

    _upload(file=file, project=project)


@app.command("list")
def list_datasets(
    project: str | None = typer.Option(
        None, "--project", help="Project name (default: current project)."
    ),
) -> None:
    """List datasets."""
    from osmosis_ai.platform.cli.dataset import list_datasets as _list_datasets

    _list_datasets(project=project)


@app.command("status")
def status(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Check dataset processing status."""
    from osmosis_ai.platform.cli.dataset import status as _status

    _status(id=id, project=project)


@app.command("preview")
def preview(
    id: str = typer.Argument(
        ..., help="Dataset ID (or short prefix from 'dataset list')."
    ),
    rows: int = typer.Option(5, "--rows", help="Number of rows to show."),
    project: str | None = typer.Option(
        None, "--project", help="Project name (used for short ID lookup)."
    ),
) -> None:
    """Preview dataset rows."""
    from osmosis_ai.platform.cli.dataset import preview as _preview

    _preview(id=id, rows=rows, project=project)


@app.command("validate")
def validate(
    file: str = typer.Argument(..., help="Path to the file to validate."),
) -> None:
    """Validate a dataset file locally."""
    from osmosis_ai.platform.cli.dataset import validate as _validate

    _validate(file=file)


@app.command("delete")
def delete() -> None:
    """Delete a dataset."""
    not_implemented("dataset", "delete")
