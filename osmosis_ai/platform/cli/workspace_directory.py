"""Local workspace directory commands for the active Git worktree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError


def _optional_git_context(workspace_directory: Path) -> dict[str, str | None]:
    from osmosis_ai.platform.cli.workspace_repo import (
        get_local_git_remote_url,
        normalize_git_identity,
    )

    remote_url = get_local_git_remote_url(workspace_directory)
    if remote_url is None:
        return {"identity": None, "remote_url": None}
    try:
        normalized = normalize_git_identity(remote_url)
    except CLIError as exc:
        return {
            "identity": None,
            "remote_url": None,
            "warning": str(exc),
        }
    return {
        "identity": normalized.identity,
        "remote_url": normalized.display_url,
    }


def doctor_workspace_directory(path: Any | None = None, *, fix: bool = False) -> Any:
    """Inspect and optionally repair the canonical workspace directory scaffold."""
    from osmosis_ai.cli.output import OperationResult
    from osmosis_ai.platform.cli.scaffold import (
        official_scaffold_updates,
        write_scaffold,
    )
    from osmosis_ai.platform.cli.workspace_directory_contract import (
        resolve_workspace_directory,
    )

    workspace_directory = resolve_workspace_directory(path)
    git = _optional_git_context(workspace_directory)
    missing = _missing_scaffold_paths(workspace_directory)

    if fix:
        write_scaffold(workspace_directory, workspace_directory.name)
        missing = _missing_scaffold_paths(workspace_directory)

    updates_available = official_scaffold_updates(workspace_directory) if fix else []

    return OperationResult(
        operation="doctor",
        status="success",
        resource={
            "workspace_directory": str(workspace_directory),
            "git": git,
            "required_paths": _required_workspace_paths(),
            "missing": missing,
            "valid": not missing,
            "updates_available": updates_available,
            "updates_checked": fix,
            "fixed": fix,
        },
        message="Workspace doctor completed.",
        display_next_steps=_doctor_display_lines(
            workspace_directory=workspace_directory,
            missing=missing,
            updates_available=updates_available,
            fixed=fix,
        ),
    )


def _doctor_display_lines(
    *,
    workspace_directory: Path,
    missing: list[str],
    updates_available: list[str],
    fixed: bool,
) -> list[str]:
    lines = [f"Workspace directory: {workspace_directory}"]
    if missing:
        lines.append("Missing scaffold paths:")
        lines.extend(f"  - {path}" for path in missing)
        if not fixed:
            lines.append("Run `osmosis doctor --fix` to create missing scaffold paths.")
    else:
        lines.append("No missing scaffold paths.")

    if updates_available:
        lines.append(
            f"Official scaffold updates available for: {', '.join(updates_available)}"
        )
        lines.append("Review local edits before replacing official scaffold files.")
    return lines


def _missing_scaffold_paths(workspace_directory: Path) -> list[str]:
    from osmosis_ai.platform.cli.scaffold import load_scaffold_entries

    scaffold, _agent_refresh_paths = load_scaffold_entries()
    missing: list[str] = []
    for entry in scaffold:
        rel_path = Path(entry.dest)
        if rel_path.name == ".gitkeep":
            directory = rel_path.parent
            if not (workspace_directory / directory).is_dir():
                missing.append(f"{directory.as_posix()}/")
            continue
        if not (workspace_directory / entry.dest).exists():
            missing.append(entry.dest)
    return missing


def _required_workspace_paths() -> list[str]:
    from osmosis_ai.templates.catalog import required_workspace_paths

    return list(required_workspace_paths())


__all__ = [
    "doctor_workspace_directory",
]
