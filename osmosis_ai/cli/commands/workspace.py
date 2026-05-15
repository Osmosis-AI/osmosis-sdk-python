"""Local workspace directory management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage the local Osmosis workspace directory.",
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


@app.command("refresh-agents")
def refresh_agents(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite local edits in official agent scaffold files.",
    ),
) -> Any:
    """Refresh official agent scaffold files after reviewing local edits."""
    from osmosis_ai.platform.cli.workspace_directory import (
        refresh_workspace_agent_files,
    )

    return refresh_workspace_agent_files(force=force)
