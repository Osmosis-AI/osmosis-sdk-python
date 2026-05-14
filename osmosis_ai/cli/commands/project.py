"""Local project management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage the local Osmosis project.",
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


@app.command("doctor")
def doctor(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Create missing scaffold paths. Existing files are never overwritten.",
    ),
) -> Any:
    """Inspect and optionally repair the canonical project scaffold."""
    from osmosis_ai.platform.cli.project import doctor_project

    return doctor_project(fix=fix)


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
    from osmosis_ai.platform.cli.project import refresh_agent_files

    return refresh_agent_files(force=force)


@app.command("link")
def link(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace ID or name to link to.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm link without prompting.",
    ),
) -> Any:
    """Link the current project to an Osmosis workspace."""
    from osmosis_ai.platform.cli.project import link_project

    return link_project(workspace=workspace, yes=yes)


@app.command("unlink")
def unlink(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm unlink without prompting.",
    ),
) -> Any:
    """Unlink the current project from its Osmosis workspace."""
    from osmosis_ai.platform.cli.project import unlink_project

    return unlink_project(yes=yes)
