from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.platform.auth.config import PLATFORM_URL

from .project_contract import resolve_project_root_from_cwd, validate_project_contract
from .project_mapping import CONFIG_FILE, ProjectMappingStore


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    project_root: Path
    workspace_id: str
    workspace_name: str
    repo_url: str | None
    credentials: Any


class WorkspaceRefResolutionError(CLIError):
    pass


def resolve_workspace_ref(ref: str, workspaces: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = [ws for ws in workspaces if ws.get("id") == ref]
    if len(by_id) == 1:
        return by_id[0]
    by_name = [ws for ws in workspaces if ws.get("name") == ref]
    if len(by_name) == 1:
        return by_name[0]
    by_name_folded = [
        ws
        for ws in workspaces
        if isinstance(ws.get("name"), str) and ws["name"].casefold() == ref.casefold()
    ]
    if len(by_name_folded) == 1:
        return by_name_folded[0]
    if len(by_id) + len(by_name) + len(by_name_folded) > 1:
        raise WorkspaceRefResolutionError(f"Workspace reference '{ref}' is ambiguous.")
    raise WorkspaceRefResolutionError(f"Workspace '{ref}' not found.")


def resolve_linked_workspace_context(*, cwd: Path | None = None) -> WorkspaceContext:
    project_root = resolve_project_root_from_cwd(cwd)
    validate_project_contract(project_root)
    store = ProjectMappingStore(config_file=CONFIG_FILE, platform_url=PLATFORM_URL)
    record = store.get_project(str(project_root))
    if record is None:
        raise CLIError(
            "This project is not linked to an Osmosis workspace for the current platform.\n"
            "Run 'osmosis project link' from the project root."
        )
    credentials = load_credentials()
    if credentials is None:
        raise CLIError("Unauthorized. Please log in with osmosis auth login")
    if credentials.is_expired():
        from osmosis_ai.platform.auth.platform_client import AuthenticationExpiredError

        raise AuthenticationExpiredError()
    return WorkspaceContext(
        project_root=project_root,
        workspace_id=record.workspace_id,
        workspace_name=record.workspace_name,
        repo_url=record.repo_url,
        credentials=credentials,
    )


def list_accessible_workspaces(*, credentials: Any) -> list[dict[str, Any]]:
    from osmosis_ai.platform.auth.platform_client import platform_request

    data = platform_request(
        "/api/cli/workspaces",
        credentials=credentials,
        require_workspace=False,
        cleanup_on_401=False,
    )
    workspaces = data.get("workspaces", [])
    if not isinstance(workspaces, list):
        raise CLIError("Invalid workspaces response from Osmosis platform.")
    if not all(isinstance(ws, dict) for ws in workspaces):
        raise CLIError("Invalid workspace entry in Osmosis platform response.")
    return workspaces


def refresh_workspace_by_id(*, workspace_id: str, credentials: Any) -> dict[str, Any]:
    for ws in list_accessible_workspaces(credentials=credentials):
        if ws.get("id") == workspace_id:
            return ws
    raise CLIError(
        "This project is linked to a workspace that is no longer accessible.\n"
        "Run 'osmosis auth login' if you logged in as a different user, or unlink "
        "and run 'osmosis project link' again."
    )
