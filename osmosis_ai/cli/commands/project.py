"""Local project management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage the local Osmosis project.",
    no_args_is_help=True,
)


@app.command("doctor")
def doctor(
    path: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        help="Path inside the project (defaults to current directory).",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Create missing scaffold paths and refresh agent scaffold files.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm scaffold refresh without prompting.",
    ),
) -> Any:
    """Inspect and optionally repair the canonical project scaffold."""
    from osmosis_ai.platform.cli.project import doctor_project

    return doctor_project(path=path, fix=fix, yes=yes)
