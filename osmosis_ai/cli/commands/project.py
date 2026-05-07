"""Local project management commands.

A "project" is the on-disk directory created by ``osmosis init`` —
distinct from a platform workspace linked via ``osmosis project link``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

app: typer.Typer = typer.Typer(
    help="Manage the local Osmosis project.",
    no_args_is_help=True,
)


@app.command("init")
def init(
    name: str = typer.Argument(
        ..., help="Project name (used for directory and config)."
    ),
    here: bool = typer.Option(
        False,
        "--here",
        help="Initialize in current directory instead of creating a subdirectory.",
    ),
) -> Any:
    """Initialize a new Osmosis project directory."""
    from osmosis_ai.platform.cli.init import init as do_init

    return do_init(name=name, here=here)


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

    return doctor_project(fix=fix, yes=yes)


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


@app.command("info")
def info(
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Refresh cached workspace metadata from the platform when linked.",
    ),
) -> Any:
    """Show project link information."""
    from osmosis_ai.platform.cli.project import project_info

    return project_info(refresh=refresh)


@app.command("list")
def list_(
    all_platforms: bool = typer.Option(
        False,
        "--all-platforms",
        help="Include linked projects for every platform URL in local config.",
    ),
) -> Any:
    """List local project-to-workspace mappings."""
    from osmosis_ai.platform.cli.project import list_projects

    return list_projects(all_platforms=all_platforms)
