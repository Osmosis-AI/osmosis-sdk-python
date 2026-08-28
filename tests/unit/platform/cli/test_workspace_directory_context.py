from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.cli import workspace_directory_context
from osmosis_ai.platform.workspace_scope import override_workspace_name


def _make_credentials(*, expired: bool = False) -> Credentials:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    return Credentials(
        access_token="token",
        token_type="Bearer",
        expires_at=now - timedelta(days=1) if expired else now + timedelta(days=1),
        user=UserInfo(id="user_1", email="user@example.com", name="User"),
        created_at=now,
    )


def _repo(path: Path, *, origin: str | None = None) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], check=True, capture_output=True
    )
    if origin is not None:
        subprocess.run(
            ["git", "-C", str(path), "remote", "add", "origin", origin],
            check=True,
            capture_output=True,
        )


def _scaffold(path: Path) -> None:
    for rel in ("rollouts", "configs/training", "configs/eval", "data"):
        (path / rel).mkdir(parents=True, exist_ok=True)


def test_optional_git_identity_resolves_origin(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://github.com/Acme/Rollouts.git")

    identity = workspace_directory_context.resolve_optional_git_identity(tmp_path)

    assert identity == "acme/rollouts"


def test_optional_git_identity_none_without_origin(tmp_path: Path) -> None:
    _repo(tmp_path)

    assert workspace_directory_context.resolve_optional_git_identity(tmp_path) is None


def test_optional_git_identity_none_outside_git_worktree(tmp_path: Path) -> None:
    assert workspace_directory_context.resolve_optional_git_identity(tmp_path) is None


def test_optional_git_identity_ignores_invalid_origin(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://gitlab.com/acme/rollouts.git")

    assert workspace_directory_context.resolve_optional_git_identity(tmp_path) is None


def test_optional_git_identity_ignores_malformed_origin(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://[github.com/acme/rollouts.git")

    assert workspace_directory_context.resolve_optional_git_identity(tmp_path) is None


def test_platform_context_requires_origin_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/Acme/Rollouts.git")
    _scaffold(tmp_path)
    remote_calls: list[Path] = []

    def get_remote(workspace_directory: Path) -> str:
        remote_calls.append(workspace_directory)
        return "https://github.com/Acme/Rollouts.git"

    monkeypatch.setattr(
        workspace_directory_context, "get_local_git_remote_url", get_remote
    )
    monkeypatch.setattr(
        workspace_directory_context, "load_credentials", lambda: _make_credentials()
    )

    ctx = workspace_directory_context.resolve_git_workspace_directory_context(
        cwd=tmp_path
    )

    assert ctx.workspace_directory == tmp_path.resolve()
    assert ctx.git_identity == "acme/rollouts"
    assert ctx.repo_url == "https://github.com/Acme/Rollouts.git"
    assert ctx.credentials.access_token == "token"
    assert remote_calls == [tmp_path.resolve()]


def test_platform_context_rejects_missing_origin(tmp_path: Path) -> None:
    _repo(tmp_path)
    _scaffold(tmp_path)

    with pytest.raises(CLIError) as exc:
        workspace_directory_context.resolve_git_workspace_directory_context(
            cwd=tmp_path
        )

    assert "Set `origin` to the Platform-connected repository" in str(exc.value)


def test_platform_context_requires_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(workspace_directory_context, "load_credentials", lambda: None)

    with pytest.raises(CLIError, match="Not logged in") as exc_info:
        workspace_directory_context.resolve_git_workspace_directory_context(
            cwd=tmp_path
        )
    assert exc_info.value.code == "AUTH_REQUIRED"


def test_platform_context_rejects_expired_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(
        workspace_directory_context,
        "load_credentials",
        lambda: _make_credentials(expired=True),
    )

    with pytest.raises(AuthenticationExpiredError):
        workspace_directory_context.resolve_git_workspace_directory_context(
            cwd=tmp_path
        )


def test_explicit_workspace_context_skips_local_git_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workspace_directory_context, "load_credentials", lambda: _make_credentials()
    )
    monkeypatch.setattr(
        workspace_directory_context,
        "resolve_git_workspace_directory_context",
        lambda **kwargs: pytest.fail("explicit workspace inspected local Git"),
    )

    with override_workspace_name("acme"):
        ctx = workspace_directory_context.resolve_platform_workspace_context(
            cwd=tmp_path
        )

    assert ctx.workspace_name == "acme"
    assert ctx.workspace_directory is None
    assert ctx.git_identity is None
    assert workspace_directory_context.workspace_result_context(ctx) == {
        "workspace": {"name": "acme"}
    }


def test_local_git_context_does_not_load_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(
        workspace_directory_context,
        "load_credentials",
        lambda: pytest.fail("local-only context loaded credentials"),
    )

    ctx = workspace_directory_context.resolve_local_workspace_directory_context(
        cwd=tmp_path
    )

    assert ctx.workspace_directory == tmp_path.resolve()
    assert ctx.git_identity == "acme/rollouts"


def test_local_workspace_context_does_not_require_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path)
    _scaffold(tmp_path)
    monkeypatch.setattr(
        workspace_directory_context,
        "load_credentials",
        lambda: pytest.fail("local-only context loaded credentials"),
    )

    ctx = workspace_directory_context.resolve_local_workspace_directory_context(
        cwd=tmp_path
    )

    assert ctx.workspace_directory == tmp_path.resolve()
    assert ctx.git_identity is None
    assert workspace_directory_context.local_result_context(ctx) == {
        "workspace_directory": str(tmp_path.resolve())
    }


def test_git_result_context_shape(tmp_path: Path) -> None:
    ctx = workspace_directory_context.GitWorkspaceDirectoryContext(
        workspace_directory=tmp_path.resolve(),
        git_identity="acme/rollouts",
        repo_url="https://github.com/acme/rollouts.git",
        credentials=_make_credentials(),
    )

    result: dict[str, Any] = workspace_directory_context.git_result_context(ctx)

    assert result == {
        "git": {
            "identity": "acme/rollouts",
            "remote_url": "https://github.com/acme/rollouts.git",
        },
        "workspace_directory": str(tmp_path.resolve()),
    }
