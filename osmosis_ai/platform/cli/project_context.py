from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError, load_credentials

from .project_contract import resolve_project_root, validate_project_contract
from .workspace_repo import get_local_git_remote_url, normalize_git_identity

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


@dataclass(frozen=True, slots=True)
class GitProjectContext:
    project_root: Path
    git_identity: str
    repo_url: str | None
    credentials: Credentials


@dataclass(frozen=True, slots=True)
class LocalProjectContext:
    project_root: Path
    git_identity: str | None
    repo_url: str | None


def _optional_identity(project_root: Path) -> tuple[str | None, str | None]:
    remote_url = get_local_git_remote_url(project_root)
    if remote_url is None:
        return None, None
    try:
        normalized = normalize_git_identity(remote_url)
    except CLIError:
        return None, None
    return normalized.identity, normalized.display_url


def resolve_local_project_context(
    *,
    cwd: Path | None = None,
    require_scaffold: bool = True,
) -> LocalProjectContext:
    project_root = resolve_project_root(cwd)
    if require_scaffold:
        validate_project_contract(project_root)
    git_identity, repo_url = _optional_identity(project_root)
    return LocalProjectContext(
        project_root=project_root,
        git_identity=git_identity,
        repo_url=repo_url,
    )


def resolve_git_project_context(*, cwd: Path | None = None) -> GitProjectContext:
    project_root = resolve_project_root(cwd)
    validate_project_contract(project_root)

    remote_url = get_local_git_remote_url(project_root)
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

    return GitProjectContext(
        project_root=project_root,
        git_identity=normalized.identity,
        repo_url=normalized.display_url,
        credentials=credentials,
    )


def git_result_context(
    ctx: GitProjectContext | LocalProjectContext,
) -> dict[str, object]:
    return {
        "git": {"identity": ctx.git_identity, "remote_url": ctx.repo_url},
        "project_root": str(ctx.project_root),
    }


__all__ = [
    "GitProjectContext",
    "LocalProjectContext",
    "git_result_context",
    "resolve_git_project_context",
    "resolve_local_project_context",
]
