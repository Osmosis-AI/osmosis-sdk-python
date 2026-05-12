"""Local project commands for the active Git worktree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.console import console
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


def validate_project(path: Any) -> Any:
    """Validate the active Git worktree's Osmosis scaffold."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.platform.cli.project_contract import (
        missing_project_paths,
        resolve_project_root,
    )

    output = get_output_context()
    project_root = resolve_project_root(path)
    git = _optional_git_context(project_root)
    missing_paths = missing_project_paths(project_root)
    data = {
        "project_root": str(project_root),
        "git": git,
        "required_paths": _REQUIRED_PATHS,
        "missing_paths": missing_paths,
        "valid": not missing_paths,
    }
    if output.format is OutputFormat.rich:
        console.table(
            [
                ("Project root", console.format_text(project_root)),
                ("Git identity", git["identity"] or "-"),
                ("Git remote URL", git["remote_url"] or "-"),
                ("Required paths", ", ".join(_REQUIRED_PATHS)),
                ("Missing paths", ", ".join(missing_paths) if missing_paths else "-"),
                ("Valid", "yes" if data["valid"] else "no"),
            ],
            title="Osmosis Git Repository Scaffold",
        )
        if missing_paths:
            console.print("Missing scaffold paths:", style="yellow")
            for path in missing_paths:
                console.print(f"  - {path}", style="yellow")
            console.print("Run `osmosis project doctor --fix` to restore them.")
        else:
            console.print("Osmosis Git repository scaffold is valid.", style="green")
        return None

    return DetailResult(
        title="Osmosis Git Repository Scaffold",
        data=data,
        fields=[
            DetailField(label="Project root", value=str(project_root)),
            DetailField(label="Git identity", value=git["identity"] or "-"),
            DetailField(label="Git remote URL", value=git["remote_url"] or "-"),
            DetailField(label="Required paths", value=", ".join(_REQUIRED_PATHS)),
            DetailField(
                label="Missing paths",
                value=", ".join(missing_paths) if missing_paths else "-",
            ),
            DetailField(label="Valid", value="yes" if data["valid"] else "no"),
        ],
    )


def doctor_project(*, fix: bool = False, yes: bool = False) -> Any:
    """Inspect and optionally repair the canonical project scaffold."""
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
    from osmosis_ai.cli.prompts import confirm
    from osmosis_ai.platform.cli.project_contract import (
        missing_project_paths,
        resolve_project_root_from_cwd,
    )
    from osmosis_ai.platform.cli.scaffold import (
        AGENT_REFRESH_PATHS,
        write_scaffold,
    )

    project_root = resolve_project_root_from_cwd()
    missing = missing_project_paths(project_root)
    refreshed = [
        path for path in sorted(AGENT_REFRESH_PATHS) if (project_root / path).exists()
    ]

    if fix:
        output = get_output_context()
        if (
            refreshed
            and not yes
            and (output.format is not OutputFormat.rich or not output.interactive)
        ):
            raise CLIError(
                "Use --yes to refresh agent scaffold in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
            )
        refresh_agents = bool(refreshed)
        if (
            refreshed
            and not yes
            and not confirm("Refresh agent scaffold files?", default=False)
        ):
            refreshed = []
            refresh_agents = False
        write_scaffold(project_root, project_root.name, update=refresh_agents)
        missing = missing_project_paths(project_root)

    return OperationResult(
        operation="project.doctor",
        status="success",
        resource={
            "project_root": str(project_root),
            "missing": missing,
            "refreshed": refreshed if fix else [],
            "fixed": fix,
        },
        message="Project doctor completed.",
    )


__all__ = [
    "doctor_project",
    "validate_project",
]
