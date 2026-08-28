from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError, load_credentials

from .workspace_directory_contract import (
    find_workspace_directory,
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
    """Local source context that deliberately does not initialize credentials."""

    workspace_directory: Path
    git_identity: str | None
    repo_url: str | None


@dataclass(frozen=True, slots=True)
class PlatformWorkspaceContext:
    """Authenticated scope for commands that do not require local rollout source."""

    credentials: Credentials
    workspace_name: str | None
    workspace_directory: Path | None
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


def resolve_optional_git_identity(cwd: Path | None = None) -> str | None:
    """Best-effort Git identity for workspace-scoped headers.

    Returns None outside a Git worktree or when `origin` cannot be
    normalized — never raises, so auth-only commands stay usable anywhere.
    """
    workspace_directory = find_workspace_directory(cwd or Path.cwd())
    if workspace_directory is None:
        return None
    identity, _repo_url = _optional_identity(workspace_directory)
    return identity


def _require_credentials() -> Credentials:
    credentials = load_credentials()
    if credentials is None:
        from osmosis_ai.platform.constants import MSG_NOT_LOGGED_IN

        raise CLIError(MSG_NOT_LOGGED_IN, code="AUTH_REQUIRED")
    if credentials.is_expired():
        raise AuthenticationExpiredError()
    return credentials


def resolve_local_workspace_directory_context(
    *, cwd: Path | None = None
) -> LocalWorkspaceDirectoryContext:
    """Resolve local source without requiring credentials or a remote origin."""
    workspace_directory = resolve_workspace_directory(cwd)
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
    local = resolve_local_workspace_directory_context(cwd=cwd)
    if local.git_identity is None or local.repo_url is None:
        raise CLIError(
            "Set `origin` to the Platform-connected repository, or clone the repository from Platform."
        )
    credentials = _require_credentials()

    return GitWorkspaceDirectoryContext(
        workspace_directory=local.workspace_directory,
        git_identity=local.git_identity,
        repo_url=local.repo_url,
        credentials=credentials,
    )


def resolve_platform_workspace_context(
    *, cwd: Path | None = None
) -> PlatformWorkspaceContext:
    """Resolve explicit workspace scope first, with local Git as the fallback."""
    from osmosis_ai.platform.workspace_scope import get_workspace_name

    workspace_name = get_workspace_name()
    if workspace_name is not None:
        return PlatformWorkspaceContext(
            credentials=_require_credentials(),
            workspace_name=workspace_name,
            workspace_directory=None,
            git_identity=None,
            repo_url=None,
        )

    git_context = resolve_git_workspace_directory_context(cwd=cwd)
    return PlatformWorkspaceContext(
        credentials=git_context.credentials,
        workspace_name=None,
        workspace_directory=git_context.workspace_directory,
        git_identity=git_context.git_identity,
        repo_url=git_context.repo_url,
    )


def git_result_context(ctx: GitWorkspaceDirectoryContext) -> dict[str, object]:
    return {
        "git": {"identity": ctx.git_identity, "remote_url": ctx.repo_url},
        "workspace_directory": str(ctx.workspace_directory),
    }


def local_result_context(ctx: LocalWorkspaceDirectoryContext) -> dict[str, object]:
    result: dict[str, object] = {"workspace_directory": str(ctx.workspace_directory)}
    if ctx.git_identity is not None:
        result["git"] = {"identity": ctx.git_identity, "remote_url": ctx.repo_url}
    return result


def workspace_result_context(ctx: PlatformWorkspaceContext) -> dict[str, object]:
    """Describe the selected scope without inventing local Git fields."""
    workspace_name = getattr(ctx, "workspace_name", None)
    if workspace_name is not None:
        return {"workspace": {"name": workspace_name}}
    if ctx.workspace_directory is None or ctx.git_identity is None:
        raise RuntimeError(
            "Platform workspace context is missing both workspace scopes"
        )
    return {
        "git": {"identity": ctx.git_identity, "remote_url": ctx.repo_url},
        "workspace_directory": str(ctx.workspace_directory),
    }


__all__ = [
    "GitWorkspaceDirectoryContext",
    "LocalWorkspaceDirectoryContext",
    "PlatformWorkspaceContext",
    "git_result_context",
    "local_result_context",
    "resolve_git_workspace_directory_context",
    "resolve_local_workspace_directory_context",
    "resolve_optional_git_identity",
    "resolve_platform_workspace_context",
    "workspace_result_context",
]
