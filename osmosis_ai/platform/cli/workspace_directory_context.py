from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError, load_credentials

from .workspace_directory_contract import (
    resolve_workspace_directory,
    validate_workspace_directory_contract,
)
from .workspace_repo import get_local_git_remote_url, normalize_git_identity

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


@dataclass(frozen=True, slots=True)
class GitWorkspaceDirectoryContext:
    workspace_directory: Path
    git_identity: str
    repo_url: str | None
    credentials: Credentials


@dataclass(frozen=True, slots=True)
class LocalWorkspaceDirectoryContext:
    workspace_directory: Path
    git_identity: str | None
    repo_url: str | None


def _optional_identity(workspace_directory: Path) -> tuple[str | None, str | None]:
    remote_url = get_local_git_remote_url(workspace_directory)
    if remote_url is None:
        return None, None
    try:
        normalized = normalize_git_identity(remote_url)
    except CLIError:
        return None, None
    return normalized.identity, normalized.display_url


def resolve_local_workspace_directory_context(
    *,
    cwd: Path | None = None,
    require_scaffold: bool = True,
) -> LocalWorkspaceDirectoryContext:
    workspace_directory = resolve_workspace_directory(cwd)
    if require_scaffold:
        validate_workspace_directory_contract(workspace_directory)
    git_identity, repo_url = _optional_identity(workspace_directory)
    return LocalWorkspaceDirectoryContext(
        workspace_directory=workspace_directory,
        git_identity=git_identity,
        repo_url=repo_url,
    )


def resolve_git_workspace_directory_context(
    *, cwd: Path | None = None
) -> GitWorkspaceDirectoryContext:
    workspace_directory = resolve_workspace_directory(cwd)
    validate_workspace_directory_contract(workspace_directory)

    remote_url = get_local_git_remote_url(workspace_directory)
    if remote_url is None:
        raise CLIError(
            "Set `origin` to the Platform-connected repository, or clone the repository from Platform."
        )
    normalized = normalize_git_identity(remote_url)

    credentials = load_credentials()
    if credentials is None:
        from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

        raise CLIError(MSG_NOT_LOGGED_IN)
    if credentials.is_expired():
        raise AuthenticationExpiredError()

    return GitWorkspaceDirectoryContext(
        workspace_directory=workspace_directory,
        git_identity=normalized.identity,
        repo_url=normalized.display_url,
        credentials=credentials,
    )


def git_result_context(
    ctx: GitWorkspaceDirectoryContext | LocalWorkspaceDirectoryContext,
) -> dict[str, object]:
    return {
        "git": {"identity": ctx.git_identity, "remote_url": ctx.repo_url},
        "workspace_directory": str(ctx.workspace_directory),
    }


__all__ = [
    "GitWorkspaceDirectoryContext",
    "LocalWorkspaceDirectoryContext",
    "git_result_context",
    "resolve_git_workspace_directory_context",
    "resolve_local_workspace_directory_context",
]
