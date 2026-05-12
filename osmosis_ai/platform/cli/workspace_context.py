from __future__ import annotations

from typing import Any

from osmosis_ai.cli.errors import CLIError


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
