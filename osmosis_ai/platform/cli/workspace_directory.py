"""Local workspace directory commands for the active Git worktree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError

_REQUIRED_PATHS = [
    "rollouts/",
    "configs/training/",
    "configs/eval/",
    "data/",
]


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
        operation="workspace.doctor",
        status="success",
        resource={
            "workspace_directory": str(workspace_directory),
            "git": git,
            "required_paths": _REQUIRED_PATHS,
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


def refresh_workspace_agent_files(*, force: bool = False) -> Any:
    """Refresh official agent scaffold files in the current workspace directory."""
    from osmosis_ai.cli.output import OperationResult
    from osmosis_ai.platform.cli.scaffold import refresh_agent_scaffold
    from osmosis_ai.platform.cli.workspace_directory_contract import (
        resolve_workspace_directory_from_cwd,
    )

    workspace_directory = resolve_workspace_directory_from_cwd()
    result = refresh_agent_scaffold(workspace_directory, force=force)
    return OperationResult(
        operation="workspace.refresh_agents",
        status="success",
        resource={"workspace_directory": str(workspace_directory), **result},
        message="Workspace agent scaffold refresh completed.",
        display_next_steps=_agent_refresh_display_lines(result),
    )


def _agent_refresh_display_lines(result: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    added = result.get("added", [])
    refreshed = result.get("refreshed", [])
    if added:
        lines.append(f"Added: {', '.join(added)}")
    if refreshed:
        lines.append(f"Refreshed: {', '.join(refreshed)}")
    if not lines:
        lines.append("No agent scaffold files changed.")
    return lines


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
            lines.append(
                "Run `osmosis workspace doctor --fix` to create missing scaffold paths."
            )
    else:
        lines.append("No missing scaffold paths.")

    if updates_available:
        lines.append(
            f"Official scaffold updates available for: {', '.join(updates_available)}"
        )
        lines.append(
            "Review local edits, then run `osmosis workspace refresh-agents --force` "
            "to replace official scaffold files."
        )
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


__all__ = [
    "doctor_workspace_directory",
    "refresh_workspace_agent_files",
]
