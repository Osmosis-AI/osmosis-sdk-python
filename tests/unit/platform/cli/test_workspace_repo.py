"""Tests for osmosis_ai.platform.cli.workspace_repo."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli import workspace_repo

# ---------------------------------------------------------------------------
# normalize_git_url
# ---------------------------------------------------------------------------


class TestNormalizeGitUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/acme/rollouts", "github.com/acme/rollouts"),
            ("https://github.com/acme/rollouts.git", "github.com/acme/rollouts"),
            ("https://github.com/acme/rollouts/", "github.com/acme/rollouts"),
            ("git@github.com:acme/rollouts.git", "github.com/acme/rollouts"),
            ("git@github.com:acme/rollouts", "github.com/acme/rollouts"),
            ("ssh://git@github.com/acme/rollouts.git", "github.com/acme/rollouts"),
            (
                "https://user:pat@github.com/acme/rollouts.git",
                "github.com/acme/rollouts",
            ),
            # Host comparison is case-insensitive.
            ("https://GitHub.com/acme/rollouts", "github.com/acme/rollouts"),
        ],
    )
    def test_known_forms_normalize_to_host_path(self, url: str, expected: str) -> None:
        assert workspace_repo.normalize_git_url(url) == expected

    @pytest.mark.parametrize("url", ["", None, "   ", "not a url", "https://"])
    def test_unparseable_input_returns_none(self, url: Any) -> None:
        assert workspace_repo.normalize_git_url(url) is None


# ---------------------------------------------------------------------------
# get_local_git_remote_url
# ---------------------------------------------------------------------------


class TestGetLocalGitRemoteUrl:
    @staticmethod
    def _make_dot_git(path: Path) -> None:
        """Create a placeholder ``.git`` directory so the guard short-circuit allows the call through."""
        (path / ".git").mkdir()

    def test_returns_none_when_git_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._make_dot_git(tmp_path)
        monkeypatch.setattr(workspace_repo.shutil, "which", lambda _name: None)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_returns_none_when_not_a_repo(self, tmp_path: Path) -> None:
        # Without ``.git`` we must short-circuit so ``git -C`` doesn't
        # walk up and surface an unrelated parent repo's origin.
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_returns_url_from_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._make_dot_git(tmp_path)
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )

        def _fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout="https://github.com/acme/rollouts.git\n", stderr=""
            )

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert (
            workspace_repo.get_local_git_remote_url(tmp_path)
            == "https://github.com/acme/rollouts.git"
        )

    def test_returns_none_on_git_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._make_dot_git(tmp_path)
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )

        def _fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 128, stdout="", stderr="fatal: not a git repository"
            )

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_returns_none_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._make_dot_git(tmp_path)
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )

        def _raise(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(workspace_repo.subprocess, "run", _raise)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_nested_dir_without_git_does_not_walk_up_to_parent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Nested project without its own ``.git`` must not surface the
        # parent repo's origin via ``git -C``'s parent-walk behavior.
        nested = tmp_path / "nested"
        nested.mkdir()

        def _fake_run(cmd, **kwargs):
            raise AssertionError("subprocess must not run when .git is absent")

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert workspace_repo.get_local_git_remote_url(nested) is None


# ---------------------------------------------------------------------------
# summarize_local_git_state
# ---------------------------------------------------------------------------


def _make_repo(path: Path) -> None:
    """Initialise a git repo with a deterministic identity for tests."""
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )


def _commit(path: Path, message: str = "init") -> None:
    subprocess.run(
        ["git", "-C", str(path), "commit", "--allow-empty", "-m", message],
        check=True,
        capture_output=True,
    )


class TestSummarizeLocalGitState:
    def test_returns_none_when_git_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(workspace_repo.shutil, "which", lambda _name: None)
        assert workspace_repo.summarize_local_git_state(tmp_path) is None

    def test_returns_none_when_not_a_repo(self, tmp_path: Path) -> None:
        # tmp_path has no .git/ — must short-circuit even if git would
        # otherwise walk up to a parent repo.
        assert workspace_repo.summarize_local_git_state(tmp_path) is None

    def test_clean_repo_with_commit(self, tmp_path: Path) -> None:
        _make_repo(tmp_path)
        _commit(tmp_path, "initial")

        state = workspace_repo.summarize_local_git_state(tmp_path)
        assert state is not None
        assert state.branch == "main"
        assert state.head_sha is not None
        assert len(state.head_sha) == 40
        assert state.is_dirty is False
        assert state.has_upstream is False
        assert state.ahead == 0

    def test_dirty_with_untracked_file(self, tmp_path: Path) -> None:
        _make_repo(tmp_path)
        _commit(tmp_path, "initial")
        (tmp_path / "scratch.txt").write_text("hello")

        state = workspace_repo.summarize_local_git_state(tmp_path)
        assert state is not None
        assert state.is_dirty is True

    def test_dirty_with_modified_tracked_file(self, tmp_path: Path) -> None:
        _make_repo(tmp_path)
        (tmp_path / "tracked.txt").write_text("v1")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "tracked.txt"],
            check=True,
            capture_output=True,
        )
        _commit(tmp_path, "add tracked")
        (tmp_path / "tracked.txt").write_text("v2")

        state = workspace_repo.summarize_local_git_state(tmp_path)
        assert state is not None
        assert state.is_dirty is True

    def test_ahead_of_upstream(self, tmp_path: Path) -> None:
        # Create an "upstream" repo and clone it locally so our branch
        # has a real upstream tracking ref to be ahead of.
        upstream = tmp_path / "upstream.git"
        subprocess.run(
            ["git", "init", "--bare", "-b", "main", str(upstream)],
            check=True,
            capture_output=True,
        )
        local = tmp_path / "local"
        subprocess.run(
            ["git", "clone", str(upstream), str(local)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(local), "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(local), "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        # First commit is pushed; second stays local.
        _commit(local, "first")
        subprocess.run(
            ["git", "-C", str(local), "push", "-u", "origin", "main"],
            check=True,
            capture_output=True,
        )
        _commit(local, "second")

        state = workspace_repo.summarize_local_git_state(local)
        assert state is not None
        assert state.has_upstream is True
        assert state.ahead == 1

    def test_detached_head_returns_none_branch(self, tmp_path: Path) -> None:
        _make_repo(tmp_path)
        _commit(tmp_path, "first")
        _commit(tmp_path, "second")
        head_sha = subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", "--detach", head_sha],
            check=True,
            capture_output=True,
        )

        state = workspace_repo.summarize_local_git_state(tmp_path)
        assert state is not None
        assert state.branch is None
        assert state.head_sha == head_sha


# ---------------------------------------------------------------------------
# require_git_top_level
# ---------------------------------------------------------------------------


class TestRequireGitTopLevel:
    def test_allows_project_at_git_top_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            workspace_repo,
            "git_worktree_top_level",
            lambda _root: tmp_path.resolve(),
            raising=False,
        )

        workspace_repo.require_git_top_level(
            tmp_path,
            command_label="`osmosis train submit`",
        )

    def test_rejects_project_below_git_top_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr(
            workspace_repo,
            "git_worktree_top_level",
            lambda _root: tmp_path.resolve(),
            raising=False,
        )

        with pytest.raises(CLIError) as exc:
            workspace_repo.require_git_top_level(
                project,
                command_label="`osmosis train submit`",
            )

        assert "Git worktree top-level Osmosis project" in str(exc.value)


# ---------------------------------------------------------------------------
# validate_workspace_repo
# ---------------------------------------------------------------------------


def _patch_refresh(
    monkeypatch: pytest.MonkeyPatch,
    return_value: dict[str, Any] | Exception,
    *,
    captured_kwargs: dict[str, Any] | None = None,
) -> None:
    """Replace OsmosisClient.refresh_workspace_info with a deterministic stub.

    When *captured_kwargs* is provided, the stub records the kwargs of each
    call into it so tests can assert e.g. ``cleanup_on_401`` was forwarded.
    """

    class _FakeClient:
        def refresh_workspace_info(self, **kwargs: Any) -> dict[str, Any]:
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)
            if isinstance(return_value, Exception):
                raise return_value
            return return_value

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", _FakeClient)


class TestValidateWorkspaceRepo:
    def _call(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        workspace_repo.validate_workspace_repo(
            project_root=tmp_path,
            workspace_id="ws_123",
            workspace_name="team-alpha",
            credentials=object(),  # type: ignore[arg-type]
            command_label="`osmosis train submit`",
        )

    def test_no_connected_repo_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(monkeypatch, {"found": True, "connected_repo": None})

        def _fail(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("git remote should not be checked")

        monkeypatch.setattr(workspace_repo, "get_local_git_remote_url", _fail)
        with pytest.raises(CLIError) as exc:
            self._call(monkeypatch, tmp_path)
        message = str(exc.value)
        assert "no connected repo" in message
        assert "team-alpha" in message
        assert "/team-alpha/integrations/git" in message
        assert "osmosis project unlink" in message
        assert "osmosis project link" in message
        assert "osmosis workspace switch" not in message

    def test_workspace_not_found_raises_stale_link_guidance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # When the linked workspace can't be matched, guide the user to
        # refresh the project mapping rather than configure Git Sync.
        _patch_refresh(monkeypatch, {"found": False})

        def _fail(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("git remote should not be checked")

        monkeypatch.setattr(workspace_repo, "get_local_git_remote_url", _fail)
        with pytest.raises(CLIError) as exc:
            self._call(monkeypatch, tmp_path)
        message = str(exc.value)
        assert "no longer accessible" in message
        assert "team-alpha" in message
        assert "ws_123" in message
        assert "osmosis project unlink" in message
        assert "osmosis project link" in message
        assert "no connected repo" not in message
        assert "/team-alpha/integrations/git" not in message
        assert "osmosis workspace switch" not in message

    def test_platform_unreachable_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(monkeypatch, PlatformAPIError("boom"))
        monkeypatch.setattr(
            workspace_repo,
            "get_local_git_remote_url",
            lambda _root: "ignored",
        )
        self._call(monkeypatch, tmp_path)

    def test_auth_error_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(monkeypatch, AuthenticationExpiredError())
        monkeypatch.setattr(
            workspace_repo,
            "get_local_git_remote_url",
            lambda _root: "ignored",
        )
        self._call(monkeypatch, tmp_path)

    def test_matching_remote_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(
            monkeypatch,
            {
                "found": True,
                "connected_repo": {
                    "repo_url": "https://github.com/acme/rollouts",
                },
            },
        )
        monkeypatch.setattr(
            workspace_repo,
            "get_local_git_remote_url",
            lambda _root: "git@github.com:acme/rollouts.git",
        )
        self._call(monkeypatch, tmp_path)

    def test_missing_local_remote_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(
            monkeypatch,
            {
                "found": True,
                "connected_repo": {
                    "repo_url": "https://github.com/acme/rollouts",
                },
            },
        )
        monkeypatch.setattr(
            workspace_repo, "get_local_git_remote_url", lambda _root: None
        )
        with pytest.raises(CLIError) as exc:
            self._call(monkeypatch, tmp_path)
        message = str(exc.value)
        assert "team-alpha" in message
        assert "https://github.com/acme/rollouts" in message
        assert "no `origin` remote" in message
        assert "osmosis project unlink" in message
        assert "osmosis project link" in message
        assert "osmosis workspace switch" not in message

    def test_mismatched_remote_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(
            monkeypatch,
            {
                "found": True,
                "connected_repo": {
                    "repo_url": "https://github.com/acme/rollouts",
                },
            },
        )
        monkeypatch.setattr(
            workspace_repo,
            "get_local_git_remote_url",
            lambda _root: "https://github.com/other/repo.git",
        )
        with pytest.raises(CLIError) as exc:
            self._call(monkeypatch, tmp_path)
        message = str(exc.value)
        assert "team-alpha" in message
        assert "https://github.com/acme/rollouts" in message
        assert "https://github.com/other/repo.git" in message
        assert "osmosis project unlink" in message
        assert "osmosis project link" in message
        assert "osmosis workspace switch" not in message

    def test_unparseable_platform_url_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_refresh(
            monkeypatch,
            {
                "found": True,
                "connected_repo": {"repo_url": "garbage"},
            },
        )

        def _fail(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("git remote should not be checked")

        monkeypatch.setattr(workspace_repo, "get_local_git_remote_url", _fail)
        self._call(monkeypatch, tmp_path)

    def test_disables_cleanup_on_401(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A pre-flight check must never wipe local credentials/active
        # workspace on a transient 401 — the actual submit call below
        # surfaces real auth failures with full context.
        captured: dict[str, Any] = {}
        _patch_refresh(
            monkeypatch,
            {"found": True, "connected_repo": None},
            captured_kwargs=captured,
        )

        def _fail(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("git remote should not be checked")

        monkeypatch.setattr(workspace_repo, "get_local_git_remote_url", _fail)
        with pytest.raises(CLIError):
            self._call(monkeypatch, tmp_path)

        assert captured.get("cleanup_on_401") is False
        assert captured.get("workspace_id") == "ws_123"
        assert "workspace_name" not in captured
