from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import AuthenticationExpiredError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.cli import project_context


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


def test_local_context_allows_missing_origin(tmp_path: Path) -> None:
    _repo(tmp_path)
    _scaffold(tmp_path)

    ctx = project_context.resolve_local_project_context(cwd=tmp_path)

    assert ctx.project_root == tmp_path.resolve()
    assert ctx.git_identity is None
    assert ctx.repo_url is None


def test_platform_context_requires_origin_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/Acme/Rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(
        project_context, "load_credentials", lambda: _make_credentials()
    )

    ctx = project_context.resolve_git_project_context(cwd=tmp_path)

    assert ctx.project_root == tmp_path.resolve()
    assert ctx.git_identity == "acme/rollouts"
    assert ctx.repo_url == "https://github.com/Acme/Rollouts.git"
    assert ctx.credentials.access_token == "token"


def test_platform_context_rejects_missing_origin(tmp_path: Path) -> None:
    _repo(tmp_path)
    _scaffold(tmp_path)

    with pytest.raises(CLIError) as exc:
        project_context.resolve_git_project_context(cwd=tmp_path)

    assert "Set `origin` to the Platform-connected repository" in str(exc.value)


def test_scaffold_required_context_rejects_missing_paths(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")

    with pytest.raises(CLIError) as exc:
        project_context.resolve_local_project_context(
            cwd=tmp_path, require_scaffold=True
        )

    assert "missing required Osmosis scaffold paths" in str(exc.value)


def test_local_context_allows_missing_scaffold_when_not_required(
    tmp_path: Path,
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")

    ctx = project_context.resolve_local_project_context(
        cwd=tmp_path,
        require_scaffold=False,
    )

    assert ctx.project_root == tmp_path.resolve()
    assert ctx.git_identity == "acme/rollouts"
    assert ctx.repo_url == "https://github.com/acme/rollouts.git"


def test_local_context_ignores_invalid_origin(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://gitlab.com/acme/rollouts.git")
    _scaffold(tmp_path)

    ctx = project_context.resolve_local_project_context(cwd=tmp_path)

    assert ctx.git_identity is None
    assert ctx.repo_url is None


def test_local_context_ignores_malformed_origin(tmp_path: Path) -> None:
    _repo(tmp_path, origin="https://[github.com/acme/rollouts.git")
    _scaffold(tmp_path)

    ctx = project_context.resolve_local_project_context(cwd=tmp_path)

    assert ctx.git_identity is None
    assert ctx.repo_url is None


def test_platform_context_requires_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(project_context, "load_credentials", lambda: None)

    with pytest.raises(CLIError, match="Not logged in"):
        project_context.resolve_git_project_context(cwd=tmp_path)


def test_platform_context_rejects_expired_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, origin="https://github.com/acme/rollouts.git")
    _scaffold(tmp_path)
    monkeypatch.setattr(
        project_context,
        "load_credentials",
        lambda: _make_credentials(expired=True),
    )

    with pytest.raises(AuthenticationExpiredError):
        project_context.resolve_git_project_context(cwd=tmp_path)


def test_git_result_context_shape(tmp_path: Path) -> None:
    ctx = project_context.LocalProjectContext(
        project_root=tmp_path.resolve(),
        git_identity="acme/rollouts",
        repo_url="https://github.com/acme/rollouts.git",
    )

    result: dict[str, Any] = project_context.git_result_context(ctx)

    assert result == {
        "git": {
            "identity": "acme/rollouts",
            "remote_url": "https://github.com/acme/rollouts.git",
        },
        "project_root": str(tmp_path.resolve()),
    }
