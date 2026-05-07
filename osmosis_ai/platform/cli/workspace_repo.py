"""Validate that the local project's git remote matches the linked workspace's
connected repository.

Used by commands that submit work to the platform (e.g. ``osmosis train submit``)
to guard against accidentally running from the wrong checkout when a workspace
is wired up to a specific Git Sync repo.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.auth.config import PLATFORM_URL

from .utils import platform_call

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials


# Captures host + path from common Git URL forms:
#   https://github.com/owner/repo[.git][/]
#   http(s)://user[:pat]@host/owner/repo[.git]
#   ssh://git@host/owner/repo[.git]
#   git@host:owner/repo[.git]
_GIT_URL_RE = re.compile(
    r"""^
    (?:[A-Za-z0-9+.\-]+://)?      # optional scheme (https://, ssh://, git://)
    (?:[^@/]+@)?                  # optional user@ (e.g. git@, user:pat@)
    (?P<host>[^:/]+)              # host
    [:/]                          # separator (':' for SSH-style, '/' for URL)
    (?P<path>.+?)                 # repo path (lazy)
    (?:\.git)?                    # optional .git suffix
    /?                            # optional trailing slash
    $""",
    re.VERBOSE,
)


def normalize_git_url(url: str | None) -> str | None:
    """Reduce a Git URL to ``host/owner/repo`` for fuzzy-equality comparison.

    Returns ``None`` for empty or unparseable input. The host is lowercased so
    case differences (``GitHub.com`` vs ``github.com``) don't trip the check.
    """
    if not url:
        return None

    match = _GIT_URL_RE.match(url.strip())
    if match is None:
        return None

    host = match.group("host").lower()
    path = match.group("path").strip("/")
    if not path:
        return None
    return f"{host}/{path}"


def get_local_git_remote_url(project_root: Path) -> str | None:
    """Return ``origin``'s URL for the project, or ``None`` if unavailable.

    Returns ``None`` when:
    * ``git`` is not installed on PATH;
    * the project is not a git repository;
    * there is no ``origin`` remote.
    """
    if shutil.which("git") is None:
        return None
    # Anchor to *this* project. ``git -C`` walks up parent directories
    # by default, which would otherwise pick up an unrelated parent
    # repo's ``origin`` (e.g. when the project is nested inside another
    # checkout). Mirrors the same guard in ``summarize_local_git_state``.
    if not (project_root / ".git").exists():
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    return url or None


def git_worktree_top_level(project_root: Path) -> Path | None:
    """Return the Git worktree top-level containing ``project_root``, if any."""
    if shutil.which("git") is None:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top).resolve() if top else None


def require_git_top_level(project_root: Path, command_label: str) -> None:
    top = git_worktree_top_level(project_root)
    if top != project_root.resolve():
        raise CLIError(
            f"{command_label} must be run from a Git worktree top-level Osmosis project."
        )


@dataclass(frozen=True, slots=True)
class LocalGitState:
    """Best-effort summary of a local git working tree.

    Attributes:
        branch: Current branch name, or ``None`` for a detached HEAD or
            when the lookup failed.
        head_sha: Full HEAD commit SHA, or ``None`` if unavailable.
        is_dirty: ``True`` when ``git status --porcelain`` reports any
            modified, staged, or untracked entries.
        has_upstream: ``True`` when the current branch tracks an upstream.
        ahead: Number of local commits not yet on the upstream. ``0``
            when there is no upstream or no unpushed commits.
    """

    branch: str | None
    head_sha: str | None
    is_dirty: bool
    has_upstream: bool
    ahead: int


def summarize_local_git_state(project_root: Path) -> LocalGitState | None:
    """Return a best-effort snapshot of the local git state for the project.

    Used by command flows (e.g. ``osmosis train submit``) to surface a
    "push your changes first" reminder, since the platform always
    pulls source from the workspace's connected Git remote.

    Returns ``None`` when ``git`` is not on PATH or ``project_root`` is
    not a git working tree (no ``.git`` entry); individual fields are
    safely defaulted when sub-commands fail rather than raising.
    """
    if shutil.which("git") is None:
        return None
    # Anchor to *this* project. ``git -C`` walks up parent directories
    # by default, which would otherwise pick up an unrelated parent
    # repo (e.g. when a project is nested inside another checkout).
    if not (project_root / ".git").exists():
        return None

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    head_sha = _run("rev-parse", "HEAD")
    if head_sha is None:
        return None

    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":  # detached
        branch = None

    porcelain = _run("status", "--porcelain")
    is_dirty = porcelain is not None and bool(porcelain)

    upstream = _run("rev-parse", "--abbrev-ref", "@{u}")
    has_upstream = upstream is not None and bool(upstream)

    ahead = 0
    if has_upstream:
        ahead_str = _run("rev-list", "--count", "@{u}..HEAD")
        if ahead_str and ahead_str.isdigit():
            ahead = int(ahead_str)

    return LocalGitState(
        branch=branch,
        head_sha=head_sha,
        is_dirty=is_dirty,
        has_upstream=has_upstream,
        ahead=ahead,
    )


def _git_sync_url(workspace_name: str) -> str:
    return f"{PLATFORM_URL}/{workspace_name}/integrations/git"


def _raise_no_connected_repo(workspace_name: str, command_label: str) -> None:
    raise CLIError(
        f"{command_label} requires the linked workspace to have a Git Sync "
        "connected repository (the platform pulls training code from there).\n"
        f"  Workspace '{workspace_name}' has no connected repo configured.\n"
        "\n"
        "  Connect a repo via Git Sync at:\n"
        f"    {_git_sync_url(workspace_name)}\n"
        "  Or relink this project to a workspace that already has one:\n"
        "    osmosis project unlink\n"
        "    osmosis project link"
    )


def _raise_workspace_not_found(
    workspace_id: str,
    workspace_name: str,
    command_label: str,
) -> None:
    raise CLIError(
        f"{command_label} requires a valid linked workspace.\n"
        f"  This project is linked to workspace '{workspace_name}' ({workspace_id}), "
        "but that workspace is no longer accessible.\n"
        "\n"
        "  Run 'osmosis auth login' if you logged in as a different user, "
        "or relink this project:\n"
        "    osmosis project unlink\n"
        "    osmosis project link"
    )


def validate_workspace_repo(
    *,
    project_root: Path,
    workspace_id: str,
    workspace_name: str,
    credentials: Credentials,
    command_label: str,
) -> None:
    """Ensure ``project_root`` is a clone of the linked workspace's connected repo.

    The platform pulls training code from the workspace's Git Sync repo, so a
    workspace without a connected repo cannot run a training submission at all.
    This function therefore enforces both:

    * The linked workspace has a ``connected_repo`` configured, and
    * The local project's ``origin`` remote points at that same repository.

    Raises :class:`CLIError` on any mismatch. Network/auth errors are swallowed
    so the actual submit call surfaces them with full context.
    """
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    try:
        info = platform_call(
            "Checking workspace Git Sync...",
            lambda: client.refresh_workspace_info(
                credentials=credentials,
                workspace_id=workspace_id,
                # Pre-flight check: a transient 401 must not wipe local
                # credentials/workspace state. Defer auth handling to the
                # actual submit call below.
                cleanup_on_401=False,
            ),
        )
    except (AuthenticationExpiredError, PlatformAPIError):
        # Defer to the actual submit call to surface platform/auth errors.
        return

    if info.get("found") is False:
        _raise_workspace_not_found(workspace_id, workspace_name, command_label)
        return  # pragma: no cover - _raise_workspace_not_found always raises

    connected_repo = info.get("connected_repo")
    repo_url: str | None = None
    if isinstance(connected_repo, dict):
        candidate = connected_repo.get("repo_url")
        if isinstance(candidate, str) and candidate:
            repo_url = candidate

    if repo_url is None:
        _raise_no_connected_repo(workspace_name, command_label)
        return  # pragma: no cover - _raise_no_connected_repo always raises

    expected = normalize_git_url(repo_url)
    if expected is None:
        # Platform returned a URL we can't parse; don't block on a format we
        # don't understand.
        return

    local_remote = get_local_git_remote_url(project_root)
    if local_remote is None:
        raise CLIError(
            f"{command_label} must be run from a clone of the workspace's "
            "connected Git repository.\n"
            f"  Workspace '{workspace_name}' is connected to:\n"
            f"    {repo_url}\n"
            f"  Local project at {project_root} has no `origin` remote.\n"
            "\n"
            "  Clone the connected repo:\n"
            f"    git clone {repo_url}\n"
            "  Or relink this project:\n"
            "    osmosis project unlink\n"
            "    osmosis project link"
        )

    if normalize_git_url(local_remote) != expected:
        raise CLIError(
            f"{command_label} must be run from a clone of the workspace's "
            "connected Git repository.\n"
            f"  Workspace '{workspace_name}' is connected to:\n"
            f"    {repo_url}\n"
            f"  Local `origin` remote:\n"
            f"    {local_remote}\n"
            "\n"
            "  Run from the connected repo, or relink this project:\n"
            "    osmosis project unlink\n"
            "    osmosis project link"
        )


__all__ = [
    "LocalGitState",
    "get_local_git_remote_url",
    "git_worktree_top_level",
    "normalize_git_url",
    "require_git_top_level",
    "summarize_local_git_state",
    "validate_workspace_repo",
]
