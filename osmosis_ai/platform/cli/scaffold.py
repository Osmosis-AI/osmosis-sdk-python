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
    OFFICIAL_AGENT_SCAFFOLD_PATHS,
    REQUIRED_SCAFFOLD_DIRS,
    ScaffoldEntry,
)
from osmosis_ai.templates.source import workspace_template_root

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


def _read_agent_scaffold_files(*, refresh_template: bool) -> dict[str, str]:
    """Read SDK-allowed agent scaffold files from the workspace template."""
    root = workspace_template_root(refresh=refresh_template)
    contents: dict[str, str] = {}
    for rel_path in OFFICIAL_AGENT_SCAFFOLD_PATHS:
        rel_path_text = rel_path.as_posix()
        source_path = root / rel_path
        if not source_path.is_file():
            raise CLIError(
                "Workspace template is missing an official agent scaffold file: "
                f"{rel_path_text}",
                code="NOT_FOUND",
            )
        contents[rel_path_text] = _render_official_scaffold(
            source_path.read_text(encoding="utf-8")
        )
    return contents


def load_scaffold_entries() -> tuple[list[ScaffoldEntry], set[str]]:
    """Load scaffold entries and official paths from the SDK catalog."""
    official_paths = {path.as_posix() for path in OFFICIAL_AGENT_SCAFFOLD_PATHS}
    entries = [
        ScaffoldEntry(
            dest=directory.joinpath(".gitkeep").as_posix(),
        )
        for directory in REQUIRED_SCAFFOLD_DIRS
    ]
    entries.extend(
        ScaffoldEntry(
            dest=path.as_posix(),
            official=True,
        )
        for path in OFFICIAL_AGENT_SCAFFOLD_PATHS
    )
    return entries, official_paths


def official_scaffold_updates(
    project_root: Path, *, refresh_template: bool = True
) -> list[str]:
    """Return official scaffold files whose local content differs from the template."""
    official_contents = _read_agent_scaffold_files(refresh_template=refresh_template)
    updates: list[str] = []
    for rel_path, official in official_contents.items():
        path = project_root / rel_path
        if not path.is_file():
            continue
        if path.read_text(encoding="utf-8") != official:
            updates.append(rel_path)
    return updates


def write_scaffold(target: Path, project_name: str, *, update: bool = False) -> None:
    """Write missing scaffold paths into *target*.

    The SDK controls the allowed paths; agent scaffold content comes from the
    latest workspace template. Existing files are never overwritten here.
    Official file updates are reported by ``official_scaffold_updates`` and can
    be applied with ``refresh_agent_scaffold``.
    """
    del project_name, update
    entries, _official_paths = load_scaffold_entries()
    missing_official = [
        entry.dest
        for entry in entries
        if entry.official and not (target / entry.dest).exists()
    ]
    official_contents = (
        _read_agent_scaffold_files(refresh_template=True) if missing_official else {}
    )
    for entry in entries:
        dest = target / entry.dest
        if dest.exists():
            continue
        content = official_contents[entry.dest] if entry.official else entry.content
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def refresh_agent_scaffold(
    project_root: Path, *, force: bool = False
) -> dict[str, list[str]]:
    """Refresh official agent scaffold files, protecting local edits by default."""
    official = _read_agent_scaffold_files(refresh_template=True)
    added: list[str] = []
    refreshed: list[str] = []
    conflicts: list[str] = []
    pending_adds: list[tuple[str, Path, str]] = []
    pending_refreshes: list[tuple[str, Path, str]] = []

    for rel_path, content in official.items():
        path = project_root / rel_path
        if not path.exists():
            pending_adds.append((rel_path, path, content))
            continue
        if path.read_text(encoding="utf-8") == content:
            continue
        if not force:
            conflicts.append(rel_path)
            continue
        pending_refreshes.append((rel_path, path, content))

    if conflicts:
        listing = "\n  ".join(conflicts)
        raise CLIError(
            "Refusing to overwrite local edits in official scaffold files:\n"
            f"  {listing}\n"
            "\nRe-run with `osmosis project refresh-agents --force` after reviewing "
            "your local changes.",
            code="CONFLICT",
        )
    for rel_path, path, content in pending_adds:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        added.append(rel_path)
    for rel_path, path, content in pending_refreshes:
        path.write_text(content, encoding="utf-8")
        refreshed.append(rel_path)
    return {"added": added, "refreshed": refreshed}


__all__ = [
    "ScaffoldEntry",
    "load_scaffold_entries",
    "official_scaffold_updates",
    "refresh_agent_scaffold",
    "write_scaffold",
]
