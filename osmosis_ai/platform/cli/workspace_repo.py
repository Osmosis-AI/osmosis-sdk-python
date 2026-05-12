"""Helpers for reading and normalizing local Git repository state."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

from osmosis_ai.cli.errors import CLIError

_SCP_LIKE_SSH_RE = re.compile(r"^(?P<user>[^@\s/]+)@(?P<host>[^:\s/]+):(?P<path>.+)$")
_GITHUB_REPOSITORY_IDENTITY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class GitRemoteIdentity:
    """Normalized GitHub repository identity and safe display URL."""

    identity: str
    display_url: str


def normalize_git_identity(url: str | None) -> GitRemoteIdentity:
    """Return the GitHub ``owner/repo`` identity for a remote URL.

    The identity is intentionally hostless and credential-free so it can be used
    in API headers or payloads without leaking the raw remote URL.
    """
    if url is None or not isinstance(url, str) or not url:
        raise CLIError("Git remote URL must be a GitHub repository URL.")
    if _has_whitespace_or_control(url):
        raise CLIError(
            "Git remote URL must not contain whitespace or control characters."
        )

    parsed = _parse_git_remote_url(url)
    if parsed.hostname is None or parsed.hostname.lower() != "github.com":
        raise CLIError("Git remote URL must be hosted on github.com.")

    normalized_path = _normalized_github_path(parsed.path)
    segments = normalized_path.split("/")
    if len(segments) != 2:
        raise CLIError("Git remote URL must identify exactly one GitHub repository.")

    identity = f"{segments[0]}/{segments[1]}".lower()
    if _GITHUB_REPOSITORY_IDENTITY_RE.fullmatch(identity) is None:
        raise CLIError("Git remote URL contains an invalid GitHub repository name.")

    return GitRemoteIdentity(
        identity=identity,
        display_url=_display_url_without_credentials(parsed),
    )


def _parse_git_remote_url(url: str) -> SplitResult:
    scp_like = _SCP_LIKE_SSH_RE.fullmatch(url)
    if scp_like is not None:
        url = (
            f"ssh://{scp_like.group('user')}@{scp_like.group('host')}/"
            f"{scp_like.group('path')}"
        )
    try:
        return urlsplit(url)
    except ValueError as exc:
        raise CLIError("Git remote URL must be a valid URL.") from exc


def _normalized_github_path(path: str) -> str:
    lowered_path = path.lower()
    if "%2f" in lowered_path or "%5c" in lowered_path:
        raise CLIError("Git remote URL path must not contain encoded path separators.")

    normalized = path.strip("/")
    if normalized.endswith(".git"):
        normalized = normalized[: -len(".git")]
    if _has_whitespace_or_control(normalized):
        raise CLIError(
            "Git remote URL path must not contain whitespace or control characters."
        )
    return normalized


def _display_url_without_credentials(parsed: SplitResult) -> str:
    netloc = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    if parsed.scheme == "ssh" and parsed.username == "git":
        netloc = f"git@{netloc}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _has_whitespace_or_control(value: str) -> bool:
    return any(
        character.isspace() or _is_control_character(character) for character in value
    )


def _is_control_character(character: str) -> bool:
    codepoint = ord(character)
    return codepoint < 32 or codepoint == 127 or 128 <= codepoint <= 159


def get_local_git_remote_url(project_root: Path) -> str | None:
    """Return ``origin``'s URL for the project, or ``None`` if unavailable.

    Returns ``None`` when:
    * ``git`` is not installed on PATH;
    * the project is not a git repository;
    * there is no ``origin`` remote.
    """
    if shutil.which("git") is None:
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
    pulls source from Git.

    Returns ``None`` when ``git`` is not on PATH or ``project_root`` is
    not a git working tree; individual fields are safely defaulted when
    sub-commands fail rather than raising.
    """
    if shutil.which("git") is None:
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


__all__ = [
    "GitRemoteIdentity",
    "LocalGitState",
    "get_local_git_remote_url",
    "git_worktree_top_level",
    "normalize_git_identity",
    "require_git_top_level",
    "summarize_local_git_state",
]
