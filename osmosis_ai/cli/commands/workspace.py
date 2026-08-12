"""Local workspace directory health commands."""

from __future__ import annotations

from pathlib import Path

import typer

from osmosis_ai.cli.output import CommandResult


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
) -> CommandResult:
    """Inspect and optionally repair the canonical workspace directory scaffold."""
    from osmosis_ai.platform.cli.workspace_directory import doctor_workspace_directory

    return doctor_workspace_directory(path=path, fix=fix)
