"""Local project commands for the active Git worktree."""

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


def _optional_git_context(project_root: Path) -> dict[str, str | None]:
    from osmosis_ai.platform.cli.workspace_repo import (
        get_local_git_remote_url,
        normalize_git_identity,
    )

    remote_url = get_local_git_remote_url(project_root)
    if remote_url is None:
        return {"identity": None, "remote_url": None}
    try:
        normalized = normalize_git_identity(remote_url)
    except CLIError:
        return {"identity": None, "remote_url": None}
    return {
        "identity": normalized.identity,
        "remote_url": normalized.display_url,
    }


def doctor_project(path: Any | None = None, *, fix: bool = False) -> Any:
    """Inspect and optionally repair the canonical project scaffold."""
    from osmosis_ai.cli.output import OperationResult
    from osmosis_ai.platform.cli.project_contract import resolve_project_root
    from osmosis_ai.platform.cli.scaffold import (
        official_scaffold_updates,
        write_scaffold,
    )

    project_root = resolve_project_root(path)
    git = _optional_git_context(project_root)
    missing = _missing_scaffold_paths(project_root)

    if fix:
        write_scaffold(project_root, project_root.name)
        missing = _missing_scaffold_paths(project_root)

    updates_available = official_scaffold_updates(project_root) if fix else []

    return OperationResult(
        operation="project.doctor",
        status="success",
        resource={
            "project_root": str(project_root),
            "git": git,
            "required_paths": _REQUIRED_PATHS,
            "missing": missing,
            "valid": not missing,
            "updates_available": updates_available,
            "fixed": fix,
        },
        message="Project doctor completed.",
        display_next_steps=_doctor_display_lines(
            project_root=project_root,
            missing=missing,
            updates_available=updates_available,
            fixed=fix,
        ),
    )


def refresh_agent_files(*, force: bool = False) -> Any:
    """Refresh official agent scaffold files in the current project."""
    from osmosis_ai.cli.output import OperationResult
    from osmosis_ai.platform.cli.project_contract import resolve_project_root_from_cwd
    from osmosis_ai.platform.cli.scaffold import refresh_agent_scaffold

    project_root = resolve_project_root_from_cwd()
    result = refresh_agent_scaffold(project_root, force=force)
    return OperationResult(
        operation="project.refresh_agents",
        status="success",
        resource={"project_root": str(project_root), **result},
        message="Project agent scaffold refresh completed.",
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
    project_root: Path,
    missing: list[str],
    updates_available: list[str],
    fixed: bool,
) -> list[str]:
    lines = [f"Project root: {project_root}"]
    if missing:
        lines.append("Missing scaffold paths:")
        lines.extend(f"  - {path}" for path in missing)
        if not fixed:
            lines.append(
                "Run `osmosis project doctor --fix` to create missing scaffold paths."
            )
    else:
        lines.append("No missing scaffold paths.")

    if updates_available:
        lines.append(
            f"Official scaffold updates available for: {', '.join(updates_available)}"
        )
        lines.append(
            "Review local edits, then run `osmosis project refresh-agents --force` "
            "to replace official scaffold files."
        )
    return lines


def _missing_scaffold_paths(project_root: Path) -> list[str]:
    from osmosis_ai.platform.cli.scaffold import load_scaffold_entries

    scaffold, _agent_refresh_paths = load_scaffold_entries()
    missing: list[str] = []
    for entry in scaffold:
        rel_path = Path(entry.dest)
        if rel_path.name == ".gitkeep":
            directory = rel_path.parent
            if not (project_root / directory).is_dir():
                missing.append(f"{directory.as_posix()}/")
            continue
        if not (project_root / entry.dest).exists():
            missing.append(entry.dest)
    return missing


__all__ = [
    "doctor_project",
    "refresh_agent_files",
]
