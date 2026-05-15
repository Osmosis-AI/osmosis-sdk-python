"""Local workspace directory health commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer


def doctor(
    path: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        help="Path inside the workspace directory (defaults to current directory).",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Create missing scaffold paths. Existing files are never overwritten.",
    ),
) -> Any:
    """Inspect and optionally repair the canonical workspace directory scaffold."""
    from osmosis_ai.platform.cli.workspace_directory import doctor_workspace_directory

    return doctor_workspace_directory(path=path, fix=fix)
