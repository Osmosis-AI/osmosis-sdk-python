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


def _linked_workspace(git_identity: str | None) -> dict[str, str | None] | None:
    """Best-effort linked-workspace lookup; None when logged out or offline."""
    if not git_identity:
        return None
    try:
        from osmosis_ai.platform.auth import load_credentials, verify_token

        credentials = load_credentials()
        if credentials is None or credentials.is_expired():
            return None
        verified = verify_token(credentials.access_token, git_identity=git_identity)
    except Exception:
        return None
    if verified.workspace is None:
        return None
    return {
        "id": verified.workspace.id,
        "name": verified.workspace.name,
        "role": verified.workspace.role,
    }


def doctor_workspace_directory(path: Any | None = None, *, fix: bool = False) -> Any:
    """Inspect and optionally repair the canonical workspace directory scaffold."""
    from osmosis_ai.cli.output import OperationResult
    from osmosis_ai.platform.cli.scaffold import write_scaffold
    from osmosis_ai.platform.cli.workspace_directory_contract import (
        resolve_workspace_directory,
    )

    workspace_directory = resolve_workspace_directory(path)
    git = _optional_git_context(workspace_directory)
    workspace = _linked_workspace(git.get("identity"))
    missing = _missing_scaffold_paths(workspace_directory)

    if fix:
        write_scaffold(workspace_directory, workspace_directory.name)
        missing = _missing_scaffold_paths(workspace_directory)

    valid = not missing
    return OperationResult(
        operation="doctor",
        status="success" if valid else "failed",
        resource={
            "workspace_directory": str(workspace_directory),
            "git": git,
            "workspace": workspace,
            "required_paths": _required_workspace_paths(),
            "missing": missing,
            "valid": valid,
            "fixed": fix,
        },
        message=(
            "Workspace doctor completed."
            if valid
            else "Workspace doctor found missing scaffold paths."
        ),
        display_next_steps=_doctor_display_lines(
            workspace_directory=workspace_directory,
            workspace=workspace,
            missing=missing,
            fixed=fix,
        ),
        exit_code=0 if valid else 1,
    )


def _doctor_display_lines(
    *,
    workspace_directory: Path,
    workspace: dict[str, str | None] | None,
    missing: list[str],
    fixed: bool,
) -> list[str]:
    lines = [f"Workspace directory: {workspace_directory}"]
    if workspace is not None:
        lines.append(f"Linked workspace: {workspace['name']}")
    if missing:
        lines.append("Missing scaffold paths:")
        lines.extend(f"  - {path}" for path in missing)
        if not fixed:
            lines.append("Run `osmosis doctor --fix` to create missing scaffold paths.")
    else:
        lines.append("No missing scaffold paths.")
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
