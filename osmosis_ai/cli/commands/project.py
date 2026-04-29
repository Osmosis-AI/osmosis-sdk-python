"""Local project management commands.

A "project" is the on-disk directory created by ``osmosis init`` —
distinct from a platform workspace (managed via ``osmosis workspace``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage the local Osmosis project (validate canonical structure).",
    no_args_is_help=True,
)


@app.command("validate")
def validate(
    path: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Project path (defaults to current directory).",
    ),
) -> Any:
    """Validate the canonical Osmosis project structure."""
    from osmosis_ai.platform.cli.project import validate_project

    return validate_project(path)
