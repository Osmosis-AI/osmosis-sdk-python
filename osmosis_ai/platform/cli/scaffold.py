"""SDK-backed scaffold repair for existing Osmosis projects.

This module owns the retained scaffold primitives used to repair an
already-created Osmosis project checkout. It is not a bootstrap flow: it does
not create projects, initialize git repositories, or make initial commits.
"""

from __future__ import annotations

import os
from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates.catalog import (
    OFFICIAL_SCAFFOLD_FILES,
    REQUIRED_SCAFFOLD_DIRS,
    ScaffoldEntry,
    official_files_by_path,
)

_PLUGIN_REPO_DEFAULT = "Osmosis-AI/osmosis-plugins"
_PLUGIN_MARKETPLACE_DEFAULT = "osmosis"


def _plugin_repo() -> str:
    """GitHub repo (`owner/name`) hosting the Osmosis plugin marketplace."""
    return os.environ.get("OSMOSIS_PLUGIN_REPO") or _PLUGIN_REPO_DEFAULT


def _plugin_marketplace() -> str:
    """Marketplace name as declared in the plugin repo's `marketplace.json`."""
    return os.environ.get("OSMOSIS_PLUGIN_MARKETPLACE") or _PLUGIN_MARKETPLACE_DEFAULT


def _render_official_scaffold(text: str) -> str:
    """Apply environment-driven plugin substitutions to official scaffold files."""
    plugin_repo = _plugin_repo()
    plugin_marketplace = _plugin_marketplace()
    return (
        text.replace(_PLUGIN_REPO_DEFAULT, plugin_repo)
        .replace(
            f"osmosis@{_PLUGIN_MARKETPLACE_DEFAULT}", f"osmosis@{plugin_marketplace}"
        )
        .replace(f'"{_PLUGIN_MARKETPLACE_DEFAULT}"', f'"{plugin_marketplace}"')
        .replace(
            f"install {_PLUGIN_MARKETPLACE_DEFAULT}", f"install {plugin_marketplace}"
        )
    )


def load_scaffold_entries() -> tuple[list[ScaffoldEntry], set[str]]:
    """Load scaffold entries and official paths from the SDK catalog."""
    official_paths = {file.path.as_posix() for file in OFFICIAL_SCAFFOLD_FILES}
    entries = [
        ScaffoldEntry(
            dest=directory.joinpath(".gitkeep").as_posix(),
        )
        for directory in REQUIRED_SCAFFOLD_DIRS
    ]
    entries.extend(
        ScaffoldEntry(
            dest=file.path.as_posix(),
            content=_render_official_scaffold(file.content),
            official=True,
        )
        for file in OFFICIAL_SCAFFOLD_FILES
    )
    return entries, official_paths


def official_scaffold_updates(project_root: Path) -> list[str]:
    """Return official scaffold files whose local content differs from the SDK."""
    updates: list[str] = []
    for file in OFFICIAL_SCAFFOLD_FILES:
        path = project_root / file.path
        if not path.is_file():
            continue
        official = _render_official_scaffold(file.content)
        if path.read_text(encoding="utf-8") != official:
            updates.append(file.path.as_posix())
    return updates


def write_scaffold(target: Path, project_name: str, *, update: bool = False) -> None:
    """Write missing scaffold files into *target* from the SDK catalog.

    Existing files are never overwritten here. Official file updates are reported
    by ``official_scaffold_updates`` and can be applied with
    ``refresh_agent_scaffold``.
    """
    del project_name, update
    entries, _official_paths = load_scaffold_entries()
    for entry in entries:
        dest = target / entry.dest
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(entry.content, encoding="utf-8")


def refresh_agent_scaffold(
    project_root: Path, *, force: bool = False
) -> dict[str, list[str]]:
    """Refresh official agent scaffold files, protecting local edits by default."""
    official = official_files_by_path()
    added: list[str] = []
    refreshed: list[str] = []
    conflicts: list[str] = []

    for rel_path, file in official.items():
        path = project_root / rel_path
        content = _render_official_scaffold(file.content)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            added.append(rel_path)
            continue
        if path.read_text(encoding="utf-8") == content:
            continue
        if not force:
            conflicts.append(rel_path)
            continue
        path.write_text(content, encoding="utf-8")
        refreshed.append(rel_path)

    if conflicts:
        listing = "\n  ".join(conflicts)
        raise CLIError(
            "Refusing to overwrite local edits in official scaffold files:\n"
            f"  {listing}\n"
            "\nRe-run with `osmosis project refresh-agents --force` after reviewing "
            "your local changes.",
            code="CONFLICT",
        )
    return {"added": added, "refreshed": refreshed}


__all__ = [
    "ScaffoldEntry",
    "load_scaffold_entries",
    "official_scaffold_updates",
    "refresh_agent_scaffold",
    "write_scaffold",
]
