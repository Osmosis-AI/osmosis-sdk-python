"""SDK-backed scaffold repair for existing Osmosis workspace directories.

This module owns the retained scaffold primitives used to repair an
already-created Osmosis workspace directory. It is not a bootstrap flow: it does
not create workspace directories, initialize git repositories, or make initial commits.
"""

from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates.catalog import (
    OFFICIAL_AGENT_SCAFFOLD_PATHS,
    REQUIRED_WORKSPACE_DIRS,
    ScaffoldEntry,
)
from osmosis_ai.templates.source import workspace_template_root


def _read_agent_scaffold_files(*, refresh_template: bool) -> dict[str, str]:
    """Read SDK-allowed agent scaffold files from the template source."""
    root = workspace_template_root(refresh=refresh_template)
    contents: dict[str, str] = {}
    for rel_path in OFFICIAL_AGENT_SCAFFOLD_PATHS:
        rel_path_text = rel_path.as_posix()
        source_path = root / rel_path
        if not source_path.is_file():
            raise CLIError(
                "Template source is missing an official agent scaffold file: "
                f"{rel_path_text}",
                code="NOT_FOUND",
            )
        contents[rel_path_text] = source_path.read_text(encoding="utf-8")
    return contents


def _first_symlinked_path(workspace_directory: Path, rel_path: str) -> str | None:
    current = workspace_directory
    for part in Path(rel_path).parts:
        current = current / part
        if current.is_symlink():
            return current.relative_to(workspace_directory).as_posix()
    return None


def _append_unique(paths: list[str], path: str) -> None:
    if path not in paths:
        paths.append(path)


def _raise_blocked_scaffold_paths(blocked_paths: list[str]) -> None:
    listing = "\n  ".join(blocked_paths)
    raise CLIError(
        "Refusing to follow symlinked or non-file scaffold paths:\n"
        f"  {listing}\n"
        "\nMove or replace these paths before repairing the agent scaffold.",
        code="CONFLICT",
    )


def load_scaffold_entries() -> tuple[list[ScaffoldEntry], set[str]]:
    """Load scaffold entries and official paths from the SDK catalog."""
    official_paths = {path.as_posix() for path in OFFICIAL_AGENT_SCAFFOLD_PATHS}
    entries = [
        ScaffoldEntry(
            dest=directory.joinpath(".gitkeep").as_posix(),
        )
        for directory in REQUIRED_WORKSPACE_DIRS
    ]
    entries.extend(
        ScaffoldEntry(
            dest=path.as_posix(),
            official=True,
        )
        for path in OFFICIAL_AGENT_SCAFFOLD_PATHS
    )
    return entries, official_paths


def write_scaffold(target: Path, project_name: str, *, update: bool = False) -> None:
    """Write missing scaffold paths into *target*.

    The SDK controls the allowed paths; agent scaffold content comes from the
    latest template source. Existing files are never overwritten here.
    """
    del project_name, update
    entries, _official_paths = load_scaffold_entries()
    blocked_paths: list[str] = []
    for entry in entries:
        symlink_path = _first_symlinked_path(target, entry.dest)
        if symlink_path is not None:
            _append_unique(blocked_paths, symlink_path)
    if blocked_paths:
        _raise_blocked_scaffold_paths(blocked_paths)
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
        content = official_contents[entry.dest] if entry.official else ""
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


__all__ = [
    "ScaffoldEntry",
    "load_scaffold_entries",
    "write_scaffold",
]
