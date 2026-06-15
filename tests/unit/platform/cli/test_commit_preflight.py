"""Tests for the pinned ``commit_sha`` preflight (check_pinned_commit)."""

from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.platform.cli.workspace_repo as workspace_repo
from osmosis_ai.platform.cli.workspace_repo import (
    PinnedCommitCheck,
    check_pinned_commit,
)

IDENTITY = "acme/rollouts"
SHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _patch_existence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    local: bool | None,
    remote: bool | None,
) -> None:
    monkeypatch.setattr(workspace_repo, "_local_commit_exists", lambda *_a, **_k: local)
    monkeypatch.setattr(
        workspace_repo, "_remote_commit_exists", lambda *_a, **_k: remote
    )


def _check() -> PinnedCommitCheck:
    return check_pinned_commit(
        workspace_directory=Path("/repo"), git_identity=IDENTITY, commit_sha=SHA
    )


# ---------------------------------------------------------------------------
# Happy paths: nothing actionable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("local", "remote"),
    [
        (True, True),  # exists everywhere
        (None, True),  # not fetched locally but confirmed on origin
        (True, None),  # exists locally, remote unreachable -> existing notice covers it
        (None, None),  # nothing could be determined
    ],
)
def test_check_pinned_commit_passes_without_error_or_warning(
    monkeypatch: pytest.MonkeyPatch,
    local: bool | None,
    remote: bool | None,
) -> None:
    _patch_existence(monkeypatch, local=local, remote=remote)
    result = _check()
    assert result.error is None
    assert result.warnings == ()


# ---------------------------------------------------------------------------
# Hard errors: remote authoritatively reports the commit is absent
# ---------------------------------------------------------------------------


def test_check_pinned_commit_errors_when_local_but_not_pushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_existence(monkeypatch, local=True, remote=False)
    result = _check()
    assert result.error is not None
    assert "exists locally but was not found on origin" in result.error
    assert IDENTITY in result.error


@pytest.mark.parametrize("local", [False, None])
def test_check_pinned_commit_errors_when_remote_missing(
    monkeypatch: pytest.MonkeyPatch,
    local: bool | None,
) -> None:
    _patch_existence(monkeypatch, local=local, remote=False)
    result = _check()
    assert result.error is not None
    assert "was not found on the connected repository" in result.error


# ---------------------------------------------------------------------------
# Soft warning: local miss + remote unreachable
# ---------------------------------------------------------------------------


def test_check_pinned_commit_warns_when_local_miss_and_remote_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_existence(monkeypatch, local=False, remote=None)
    result = _check()
    assert result.error is None
    assert len(result.warnings) == 1
    assert SHA in result.warnings[0]


# ---------------------------------------------------------------------------
# Low-level helper edge cases
# ---------------------------------------------------------------------------


def test_local_commit_exists_returns_none_outside_git_repo(tmp_path: Path) -> None:
    # tmp_path has no .git directory.
    assert workspace_repo._local_commit_exists(tmp_path, SHA) is None


def test_remote_commit_exists_returns_none_for_invalid_identity() -> None:
    assert workspace_repo._remote_commit_exists("not-a-valid-identity", SHA) is None
