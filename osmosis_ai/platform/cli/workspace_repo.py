"""Helpers for reading and normalizing local Git repository state."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from osmosis_ai.cli.errors import CLIError

_SCP_LIKE_SSH_RE = re.compile(r"^(?P<user>[^@\s/]+)@(?P<host>[^:\s/]+):(?P<path>.+)$")
_GITHUB_REPOSITORY_IDENTITY_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+$"
)


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

    owner, repo = segments
    identity = f"{owner}/{repo}".lower()
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


def _resolve_canonical_github_identity(owner: str, repo: str) -> str | None:
    """Resolve canonical owner/repo, best-effort for renamed repositories."""
    return _resolve_via_gh(owner, repo) or _resolve_via_git_credentials(owner, repo)


def resolve_canonical_git_identity(identity: str) -> str | None:
    """Resolve the current GitHub owner/repo for a local repository identity.

    This may contact GitHub and is intentionally kept out of normal command
    setup. Call it only after the platform rejects the local repository scope.
    """
    if _GITHUB_REPOSITORY_IDENTITY_RE.fullmatch(identity) is None:
        return None

    owner, repo = identity.split("/", maxsplit=1)
    canonical_identity = _resolve_canonical_github_identity(owner, repo)
    if (
        canonical_identity is not None
        and _GITHUB_REPOSITORY_IDENTITY_RE.fullmatch(canonical_identity) is not None
    ):
        return canonical_identity
    return None


def _resolve_via_gh(owner: str, repo: str) -> str | None:
    if shutil.which("gh") is None:
        return None
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}", "--jq", ".full_name"],
            capture_output=True,
            env=_noninteractive_git_env(),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _github_token_from_git_credentials() -> str | None:
    """Best-effort GitHub token from the local ``git credential`` helper."""
    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n",
            capture_output=True,
            env=_noninteractive_git_env(),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return _password_from_git_credentials(result.stdout)


def _resolve_via_git_credentials(owner: str, repo: str) -> str | None:
    token = _github_token_from_git_credentials()
    if token is None:
        return None

    try:
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={"Authorization": f"token {token}"},
            allow_redirects=True,
            timeout=10,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        full_name = response.json().get("full_name")
    except ValueError:
        return None
    return full_name if isinstance(full_name, str) and full_name else None


def _password_from_git_credentials(output: str) -> str | None:
    for line in output.splitlines():
        if line.startswith("password="):
            token = line.removeprefix("password=")
            return token or None
    return None


def _noninteractive_git_env() -> dict[str, str]:
    return {
        **os.environ,
        "GH_PROMPT_DISABLED": "1",
        "GCM_INTERACTIVE": "never",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _has_whitespace_or_control(value: str) -> bool:
    return any(
        character.isspace() or _is_control_character(character) for character in value
    )


def _is_control_character(character: str) -> bool:
    codepoint = ord(character)
    return codepoint < 32 or codepoint == 127 or 128 <= codepoint <= 159


def get_local_git_remote_url(workspace_directory: Path) -> str | None:
    """Return ``origin``'s URL for the workspace directory, or ``None`` if unavailable.

    Returns ``None`` when:
    * ``git`` is not installed on PATH;
    * the workspace directory is not a git repository top-level;
    * there is no ``origin`` remote.
    """
    if shutil.which("git") is None:
        return None
    if not (workspace_directory / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace_directory), "remote", "get-url", "origin"],
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


def git_worktree_top_level(workspace_directory: Path) -> Path | None:
    """Return the Git worktree top-level containing ``workspace_directory``, if any."""
    if shutil.which("git") is None:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(workspace_directory), "rev-parse", "--show-toplevel"],
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


def require_git_top_level(workspace_directory: Path, command_label: str) -> None:
    top = git_worktree_top_level(workspace_directory)
    if top != workspace_directory.resolve():
        raise CLIError(
            f"{command_label} must be run from a Git worktree top-level Osmosis workspace directory."
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


def summarize_local_git_state(workspace_directory: Path) -> LocalGitState | None:
    """Return a best-effort snapshot of the local git state for the workspace directory.

    Used by command flows (e.g. ``osmosis train submit``) to surface a
    "push your changes first" reminder, since the platform always
    pulls source from Git.

    Returns ``None`` when ``git`` is not on PATH or ``workspace_directory`` is
    not a git working tree top-level; individual fields are safely defaulted when
    sub-commands fail rather than raising.
    """
    if shutil.which("git") is None:
        return None
    if not (workspace_directory / ".git").exists():
        return None

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(workspace_directory), *args],
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


def _local_commit_exists(workspace_directory: Path, commit_sha: str) -> bool | None:
    """Return whether ``commit_sha`` resolves to a commit object in the local repo.

    Returns ``True``/``False`` when the check ran, or ``None`` when it could not
    (``git`` missing, or ``workspace_directory`` is not a git working tree).
    ``None`` is intentionally distinct from ``False`` so callers can treat
    "can't tell" differently from "definitely absent".
    """
    if shutil.which("git") is None:
        return None
    if not (workspace_directory / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(workspace_directory),
                "cat-file",
                "-e",
                f"{commit_sha}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.returncode == 0


def _commit_exists_via_gh(owner: str, repo: str, commit_sha: str) -> bool | None:
    if shutil.which("gh") is None:
        return None
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo}/commits/{commit_sha}", "--jq", ".sha"],
            capture_output=True,
            env=_noninteractive_git_env(),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode == 0:
        return True
    # GitHub returns HTTP 422 ("No commit found for SHA") for a well-formed SHA
    # that is absent. A 404 means the repo/visibility is the problem (not the
    # commit), so we treat only the 422 signal as authoritative absence.
    stderr = (result.stderr or "").lower()
    if "no commit found" in stderr or "http 422" in stderr:
        return False
    return None


def _commit_exists_via_git_credentials(
    owner: str, repo: str, commit_sha: str
) -> bool | None:
    token = _github_token_from_git_credentials()
    if token is None:
        return None
    try:
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            },
            allow_redirects=True,
            timeout=10,
        )
    except requests.RequestException:
        return None

    if response.status_code == 200:
        return True
    if response.status_code == 422:
        return False
    return None


def _remote_commit_exists(identity: str, commit_sha: str) -> bool | None:
    """Return whether ``commit_sha`` exists on the GitHub repo ``identity``.

    ``identity`` is a normalized ``owner/repo`` string. Tries the ``gh`` CLI
    first, then a token from the local git credential helper. Returns
    ``True``/``False`` only when GitHub answers authoritatively, or ``None`` when
    the check could not run (no ``gh`` and no token, or a transport/visibility
    error) so callers never block a submit on missing tooling.
    """
    if _GITHUB_REPOSITORY_IDENTITY_RE.fullmatch(identity) is None:
        return None
    owner, repo = identity.split("/", maxsplit=1)
    via_gh = _commit_exists_via_gh(owner, repo, commit_sha)
    if via_gh is not None:
        return via_gh
    return _commit_exists_via_git_credentials(owner, repo, commit_sha)


@dataclass(frozen=True, slots=True)
class PinnedCommitCheck:
    """Outcome of a pinned ``commit_sha`` preflight.

    ``error`` is set only when the commit is *confirmed* unusable (the platform
    would fail to fetch it); raising on it fails the submit fast. ``warnings``
    carries non-blocking advisories shown before the confirmation prompt.
    """

    error: str | None = None
    warnings: tuple[str, ...] = ()


def check_pinned_commit(
    *, workspace_directory: Path, git_identity: str, commit_sha: str
) -> PinnedCommitCheck:
    """Preflight a pinned ``commit_sha`` before a cloud submit.

    The platform fetches the pinned commit from the connected GitHub repository,
    so a remote answer is authoritative. The local repo is only a fast,
    offline signal — a commit may exist on origin without being fetched locally,
    so a local miss alone is never treated as an error (only a warning when the
    remote can't be reached).
    """
    local = _local_commit_exists(workspace_directory, commit_sha)
    remote = _remote_commit_exists(git_identity, commit_sha)

    if remote is False:
        if local is True:
            return PinnedCommitCheck(
                error=(
                    f"Pinned commit {commit_sha} exists locally but was not found on "
                    f"origin ({git_identity}). Push it before submitting — the platform "
                    "fetches the pinned commit from the connected repository."
                )
            )
        return PinnedCommitCheck(
            error=(
                f"Pinned commit {commit_sha} was not found on the connected repository "
                f"({git_identity}). Double-check experiment.commit_sha and make sure the "
                "commit is pushed to origin."
            )
        )

    if remote is True:
        return PinnedCommitCheck()

    # remote is None → GitHub could not confirm. Fall back to the local signal.
    if local is False:
        return PinnedCommitCheck(
            warnings=(
                f"Could not find pinned commit {commit_sha} in the local repository, "
                "and could not reach GitHub to confirm it. Make sure "
                "experiment.commit_sha is correct and pushed to origin.",
            )
        )
    return PinnedCommitCheck()


__all__ = [
    "GitRemoteIdentity",
    "LocalGitState",
    "check_pinned_commit",
    "get_local_git_remote_url",
    "git_worktree_top_level",
    "normalize_git_identity",
    "require_git_top_level",
    "resolve_canonical_git_identity",
    "summarize_local_git_state",
]
