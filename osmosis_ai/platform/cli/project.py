"""Local project commands (validate and link canonical project layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from questionary import Choice, Separator

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.cli.project_mapping import (
    CONFIG_FILE,
    ProjectLinkRecord,
    ProjectMappingEntry,
    ProjectMappingStore,
    normalize_platform_key,
    now_linked_at,
    sanitize_repo_url,
)
from osmosis_ai.platform.cli.workspace_context import (
    list_accessible_workspaces,
    refresh_workspace_by_id,
    resolve_workspace_ref,
)


def validate_project(path: Any) -> Any:
    """Validate the canonical Osmosis project structure."""
    from osmosis_ai.cli.output import (
        DetailField,
        DetailResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root,
        validate_project_contract,
    )

    output = get_output_context()
    project_root = resolve_project_root(path)
    validate_project_contract(project_root)
    rows = [
        ("Root", str(project_root)),
        ("Project metadata", ".osmosis/project.toml"),
        ("Rollouts", "rollouts/"),
        ("Training configs", "configs/training/"),
        ("Eval configs", "configs/eval/"),
        ("Datasets", "data/"),
    ]
    if output.format is OutputFormat.rich:
        console.table(
            [
                ("Root", console.format_text(project_root)),
                ("Project metadata", ".osmosis/project.toml"),
                ("Rollouts", "rollouts/"),
                ("Training configs", "configs/training/"),
                ("Eval configs", "configs/eval/"),
                ("Datasets", "data/"),
            ],
            title="Project Contract",
        )
        console.print("Project contract is valid.", style="green")
        return None

    return DetailResult(
        title="Project Contract",
        data={
            "root": str(project_root),
            "required_paths": [value for label, value in rows if label != "Root"],
            "valid": True,
        },
        fields=[DetailField(label=label, value=value) for label, value in rows],
    )


def doctor_project(*, fix: bool = False, yes: bool = False) -> Any:
    """Inspect and optionally repair the canonical project scaffold."""
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
    from osmosis_ai.cli.prompts import confirm
    from osmosis_ai.platform.cli.init import (
        AGENT_REFRESH_PATHS,
        SCAFFOLD,
        _write_scaffold,
    )
    from osmosis_ai.platform.cli.project_contract import resolve_project_root_from_cwd

    project_root = resolve_project_root_from_cwd()
    missing = _missing_scaffold_paths(project_root)
    refreshed = [
        entry.dest
        for entry in SCAFFOLD
        if entry.dest in AGENT_REFRESH_PATHS and (project_root / entry.dest).exists()
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
        _write_scaffold(project_root, project_root.name, update=refresh_agents)
        missing = _missing_scaffold_paths(project_root)

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


def _missing_scaffold_paths(project_root: Path) -> list[str]:
    from osmosis_ai.platform.cli.init import SCAFFOLD

    return [
        entry.dest for entry in SCAFFOLD if not (project_root / entry.dest).exists()
    ]


def _workspace_summary(workspace: dict[str, Any]) -> dict[str, Any]:
    workspace_id = workspace.get("id")
    workspace_name = workspace.get("name")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise CLIError("Invalid workspace record from Osmosis platform: missing id.")
    if not isinstance(workspace_name, str) or not workspace_name:
        raise CLIError("Invalid workspace record from Osmosis platform: missing name.")
    return {
        "id": workspace_id,
        "name": workspace_name,
    }


def _record_workspace(record: ProjectLinkRecord) -> dict[str, Any]:
    return {
        "id": record.workspace_id,
        "name": record.workspace_name,
        "repo_url": record.repo_url,
    }


def _link_data(
    *,
    project_root: Path,
    linked: bool,
    record: ProjectLinkRecord | None,
) -> dict[str, Any]:
    return {
        "cwd": str(Path.cwd().resolve()),
        "project_root": str(project_root),
        "platform": PLATFORM_URL,
        "platform_key": normalize_platform_key(PLATFORM_URL),
        "linked": linked,
        "workspace": _record_workspace(record) if record is not None else None,
    }


def _current_project_root() -> Path | None:
    from osmosis_ai.platform.cli.project_contract import resolve_project_root_from_cwd

    try:
        return resolve_project_root_from_cwd()
    except CLIError:
        return None


def _project_list_item(
    entry: ProjectMappingEntry, *, current_project_root: Path | None
) -> dict[str, Any]:
    record = entry.record
    project_root = Path(record.project_path)
    workspace = {
        "id": record.workspace_id,
        "name": record.workspace_name,
        "repo_url": record.repo_url,
    }
    return {
        "project_root": record.project_path,
        "platform": entry.platform_key,
        "workspace": workspace,
        "linked_at": record.linked_at,
        "exists": project_root.exists(),
        "current": (
            current_project_root is not None
            and record.project_path == str(current_project_root)
        ),
    }


def _project_list_display_item(item: dict[str, Any]) -> dict[str, Any]:
    workspace = item["workspace"]
    return {
        **item,
        "current": "*" if item["current"] else "",
        "workspace": f"{workspace['name']} ({workspace['id']})",
        "exists": "yes" if item["exists"] else "no",
    }


def _require_noninteractive_link_flags(workspace: str | None, yes: bool) -> None:
    from osmosis_ai.cli.output import OutputFormat, get_output_context

    output = get_output_context()
    if output.format is OutputFormat.rich and output.interactive:
        return
    if workspace and yes:
        return
    raise CLIError(
        "Use --workspace and --yes to link a project in non-interactive mode.",
        code="INTERACTIVE_REQUIRED",
    )


def _require_link_credentials() -> Any:
    from osmosis_ai.platform.auth.platform_client import AuthenticationExpiredError
    from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

    credentials = load_credentials()
    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN, code="AUTH_REQUIRED")
    if credentials.is_expired():
        raise AuthenticationExpiredError()
    return credentials


def _repo_url_from_workspace(workspace: dict[str, Any]) -> str | None:
    connected_repo = workspace.get("connected_repo")
    if isinstance(connected_repo, dict):
        repo_url = connected_repo.get("repo_url")
        if isinstance(repo_url, str):
            return sanitize_repo_url(repo_url)
    repo_url = workspace.get("repo_url")
    if isinstance(repo_url, str):
        return sanitize_repo_url(repo_url)
    return None


def _raw_repo_url_from_workspace(workspace: dict[str, Any]) -> str | None:
    connected_repo = workspace.get("connected_repo")
    if isinstance(connected_repo, dict):
        repo_url = connected_repo.get("repo_url")
        if isinstance(repo_url, str) and repo_url:
            return repo_url
    repo_url = workspace.get("repo_url")
    if isinstance(repo_url, str) and repo_url:
        return repo_url
    return None


def _git_sync_url(workspace_name: str) -> str:
    return f"{PLATFORM_URL}/{workspace_name}/integrations/git"


def _require_connected_repo_checkout(
    *,
    project_root: Path,
    workspace_name: str,
    workspace: dict[str, Any],
) -> None:
    from osmosis_ai.platform.cli.workspace_repo import (
        get_local_git_remote_url,
        normalize_git_url,
    )

    repo_url = _raw_repo_url_from_workspace(workspace)
    if repo_url is None:
        raise CLIError(
            f"Workspace '{workspace_name}' has no Git Sync connected repository.\n"
            "\n"
            "Connect a repo via Git Sync:\n"
            f"  {_git_sync_url(workspace_name)}\n"
            "\n"
            "Then clone it and link from that checkout:\n"
            "  git clone <repo-url>\n"
            "  cd <repo>\n"
            f"  osmosis project link --workspace {workspace_name}"
        )

    expected = normalize_git_url(repo_url)
    if expected is None:
        raise CLIError(
            "Unable to validate the workspace's Git Sync connected repository URL.\n"
            f"  Workspace '{workspace_name}' is connected to:\n"
            f"    {repo_url}\n"
            "\n"
            "Reconnect the repo in Git Sync, then run project link again:\n"
            f"  {_git_sync_url(workspace_name)}"
        )

    local_remote = get_local_git_remote_url(project_root)
    if normalize_git_url(local_remote) == expected:
        return

    if local_remote is None:
        raise CLIError(
            "Project link must be run from a clone of the workspace's "
            "connected Git repository.\n"
            f"  Workspace '{workspace_name}' is connected to:\n"
            f"    {repo_url}\n"
            f"  Local project at {project_root} has no `origin` remote.\n"
            "\n"
            "Clone the connected repo:\n"
            f"  git clone {repo_url}"
        )

    raise CLIError(
        "Project link must be run from a clone of the workspace's "
        "connected Git repository.\n"
        f"  Workspace '{workspace_name}' is connected to:\n"
        f"    {repo_url}\n"
        "  Local `origin` remote:\n"
        f"    {local_remote}\n"
        "\n"
        "Run from the connected repo checkout, then link again."
    )


def _select_workspace_interactive(
    workspaces: list[dict[str, Any]],
) -> dict[str, Any]:
    from osmosis_ai.cli.prompts import select

    if not workspaces:
        raise CLIError("No accessible Osmosis workspaces found.", code="NOT_FOUND")

    choices: list[str | Choice | Separator] = [
        Choice(
            title=str(workspace.get("name") or workspace.get("id")),
            value=workspace,
        )
        for workspace in workspaces
    ]
    selected = select("Choose a workspace to link", choices=choices)
    if selected is None:
        raise CLIError("Project link cancelled.", code="CANCELLED")
    if not isinstance(selected, dict):
        raise CLIError("Invalid workspace selection.")
    return selected


def link_project(workspace: str | None = None, yes: bool = False) -> Any:
    """Link the current project to an Osmosis workspace."""
    from osmosis_ai.cli.output import (
        OperationResult,
        OutputFormat,
        get_output_context,
    )
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root_from_cwd,
        validate_project_contract,
    )

    output = get_output_context()
    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)
    _require_noninteractive_link_flags(workspace, yes)

    credentials = _require_link_credentials()
    workspaces = list_accessible_workspaces(credentials=credentials)
    selected = (
        resolve_workspace_ref(workspace, workspaces)
        if workspace
        else _select_workspace_interactive(workspaces)
    )
    workspace_summary = _workspace_summary(selected)
    _require_connected_repo_checkout(
        project_root=project_root,
        workspace_name=workspace_summary["name"],
        workspace=selected,
    )
    record = ProjectLinkRecord(
        project_path=str(project_root),
        workspace_id=workspace_summary["id"],
        workspace_name=workspace_summary["name"],
        repo_url=_repo_url_from_workspace(selected),
        linked_at=now_linked_at(),
    )

    if not yes and output.format is OutputFormat.rich and output.interactive:
        require_confirmation(
            f"Link this project to workspace '{record.workspace_name}'?",
            yes=False,
        )

    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    stored = store.link(record)
    data = _link_data(project_root=project_root, linked=True, record=stored)
    message = f"Linked project to workspace: {stored.workspace_name}"

    if output.format is OutputFormat.rich:
        console.print(
            f"Linked project to workspace: {console.escape(stored.workspace_name)}",
            style="green",
        )
        return None

    return OperationResult(
        operation="project.link",
        status="success",
        resource=data,
        message=message,
    )


def unlink_project(yes: bool = False) -> Any:
    """Unlink the current project from its workspace."""
    from osmosis_ai.cli.output import OperationResult, OutputFormat, get_output_context
    from osmosis_ai.cli.prompts import require_confirmation
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root_from_cwd,
        validate_project_contract,
    )

    output = get_output_context()
    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)
    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    existing = store.get_project(str(project_root))
    if existing is not None and not yes:
        if output.format is not OutputFormat.rich or not output.interactive:
            raise CLIError(
                "Use --yes to unlink this project in non-interactive mode.",
                code="INTERACTIVE_REQUIRED",
            )
        require_confirmation(
            f"Unlink this project from workspace '{existing.workspace_name}'?",
            yes=False,
        )

    removed = store.unlink(str(project_root)) if existing is not None else None
    data = _link_data(project_root=project_root, linked=False, record=removed)
    message = (
        f"Unlinked project from workspace: {removed.workspace_name}"
        if removed is not None
        else "Project is not linked."
    )

    if output.format is OutputFormat.rich:
        if removed is not None:
            console.print(
                "Unlinked project from workspace: "
                f"{console.escape(removed.workspace_name)}",
                style="green",
            )
        else:
            console.print(message, style="dim")
        return None

    return OperationResult(
        operation="project.unlink",
        status="success",
        resource=data,
        message=message,
    )


def project_info(refresh: bool = False) -> Any:
    """Show project link information."""
    from osmosis_ai.cli.output import DetailField, DetailResult
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root_from_cwd,
        validate_project_contract,
    )

    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)
    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    record = store.get_project(str(project_root))
    if refresh and record is not None:
        credentials = _require_link_credentials()
        workspace = refresh_workspace_by_id(
            workspace_id=record.workspace_id,
            credentials=credentials,
        )
        workspace_summary = _workspace_summary(workspace)
        record = store.update_workspace_cache(
            str(project_root),
            workspace_name=workspace_summary["name"],
            repo_url=_repo_url_from_workspace(workspace),
        )

    data = _link_data(
        project_root=project_root, linked=record is not None, record=record
    )
    return DetailResult(
        title="Project Info",
        data=data,
        fields=[
            DetailField(label="CWD", value=data["cwd"]),
            DetailField(label="Project root", value=data["project_root"]),
            DetailField(label="Platform", value=data["platform_key"]),
            DetailField(label="Linked", value=data["linked"]),
            DetailField(
                label="Workspace",
                value=(
                    f"{record.workspace_name} ({record.workspace_id})"
                    if record is not None
                    else None
                ),
            ),
        ],
    )


def list_projects(*, all_platforms: bool = False) -> Any:
    """List local project-to-workspace mappings."""
    from osmosis_ai.cli.output import ListColumn, ListResult

    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    if all_platforms:
        entries = store.list_all_projects()
    else:
        entries = [
            ProjectMappingEntry(platform_key=store.platform_key, record=record)
            for record in store.list_projects()
        ]
    current_project_root = _current_project_root()
    items = [
        _project_list_item(entry, current_project_root=current_project_root)
        for entry in entries
    ]
    return ListResult(
        title="Linked Projects",
        items=items,
        total_count=len(items),
        has_more=False,
        next_offset=None,
        columns=[
            ListColumn(key="current", label="", no_wrap=True),
            ListColumn(key="project_root", label="Project"),
            ListColumn(key="workspace", label="Workspace"),
            ListColumn(key="platform", label="Platform", no_wrap=True),
            ListColumn(key="exists", label="Exists", no_wrap=True),
            ListColumn(key="linked_at", label="Linked At", no_wrap=True),
        ],
        display_items=[_project_list_display_item(item) for item in items],
        extra={
            "platform": PLATFORM_URL,
            "platform_key": store.platform_key,
            "all_platforms": all_platforms,
        },
    )


__all__ = [
    "CONFIG_FILE",
    "doctor_project",
    "link_project",
    "list_projects",
    "project_info",
    "unlink_project",
    "validate_project",
]
