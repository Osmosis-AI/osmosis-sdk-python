"""Tests for osmosis_ai.platform.cli.workspace_repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.cli import workspace_repo

# ---------------------------------------------------------------------------
# normalize_git_identity
# ---------------------------------------------------------------------------


class TestNormalizeGitIdentity:
    @pytest.mark.parametrize(
        ("url", "expected_identity", "expected_display"),
        [
            (
                "https://github.com/acme/rollouts.git",
                "acme/rollouts",
                "https://github.com/acme/rollouts.git",
            ),
            (
                "https://github.com/Acme/Rollouts.git",
                "acme/rollouts",
                "https://github.com/Acme/Rollouts.git",
            ),
            (
                "https://user:token@github.com/acme/rollouts.git",
                "acme/rollouts",
                "https://github.com/acme/rollouts.git",
            ),
            (
                "http://github.com/acme/rollouts",
                "acme/rollouts",
                "http://github.com/acme/rollouts",
            ),
            (
                "git://github.com/acme/rollouts.git",
                "acme/rollouts",
                "git://github.com/acme/rollouts.git",
            ),
            (
                "git@github.com:acme/rollouts.git",
                "acme/rollouts",
                "ssh://git@github.com/acme/rollouts.git",
            ),
            (
                "ssh://git@github.com/acme/rollouts.git",
                "acme/rollouts",
                "ssh://git@github.com/acme/rollouts.git",
            ),
            (
                "ssh://git@github.com/acme/repo_name.git/",
                "acme/repo_name",
                "ssh://git@github.com/acme/repo_name.git/",
            ),
        ],
    )
    def test_github_remotes_normalize_to_hostless_lowercase_identity(
        self, url: str, expected_identity: str, expected_display: str
    ) -> None:
        result = workspace_repo.normalize_git_identity(url)
        assert result.identity == expected_identity
        assert result.display_url == expected_display

    @pytest.mark.parametrize(
        "url",
        [
            "",
            None,
            "   ",
            "not a url",
            "https://",
            "/Users/me/repo",
            "../repo",
            "git@github-work:acme/rollouts.git",
            "https://gitlab.com/acme/rollouts.git",
            "https://github.com/acme/team/rollouts.git",
            "https://github.com/acme.git",
            "https://github.com/acme/has%2fslash.git",
            "https://github.com/acme/has%5cslash.git",
            "https://github.com/acme/white space.git",
            "https://github.com/-bad/rollouts.git",
            "https://github.com/acme/bad~repo.git",
        ],
    )
    def test_invalid_remote_rejected(self, url: str | None) -> None:
        with pytest.raises(CLIError):
            workspace_repo.normalize_git_identity(url)

    @pytest.mark.parametrize("control_character", ["\x7f", "\x80", "\x9f"])
    @pytest.mark.parametrize(
        "url_template",
        [
            "https://github.com/acme/rollouts.git?token={control_character}",
            "https://github.com/acme/rollouts.git#{control_character}",
            "https://user{control_character}:token@github.com/acme/rollouts.git",
            "https://github.com/acme/rollouts{control_character}.git",
            "git@github.com:acme/rollouts{control_character}.git",
        ],
    )
    def test_control_characters_rejected_anywhere_in_remote_url(
        self, control_character: str, url_template: str
    ) -> None:
        with pytest.raises(CLIError):
            workspace_repo.normalize_git_identity(
                url_template.format(control_character=control_character)
            )

    def test_header_identity_does_not_include_host_or_token(self) -> None:
        result = workspace_repo.normalize_git_identity(
            "https://token:secret@github.com/Acme/Rollouts.git"
        )
        assert result.identity == "acme/rollouts"
        assert "github.com" not in result.identity
        assert "secret" not in result.identity
        assert "secret" not in result.display_url


# ---------------------------------------------------------------------------
# get_local_git_remote_url
# ---------------------------------------------------------------------------


class TestGetLocalGitRemoteUrl:
    def test_returns_none_when_git_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(workspace_repo.shutil, "which", lambda _name: None)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_returns_none_when_not_a_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )
        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 128, stdout="", stderr="fatal: not a git repository"
            )

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None
        assert calls == [["git", "-C", str(tmp_path), "remote", "get-url", "origin"]]

    def test_returns_url_from_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )

        def _raise(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(workspace_repo.subprocess, "run", _raise)
        assert workspace_repo.get_local_git_remote_url(tmp_path) is None

    def test_nested_dir_uses_git_command_as_source_of_truth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )
        nested = tmp_path / "nested"
        nested.mkdir()
        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 128, stdout="", stderr="fatal: not a git repository"
            )

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert workspace_repo.get_local_git_remote_url(nested) is None
        assert calls == [["git", "-C", str(nested), "remote", "get-url", "origin"]]


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

    def test_returns_none_when_not_a_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            workspace_repo.shutil, "which", lambda _name: "/usr/bin/git"
        )
        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 128, stdout="", stderr="fatal: not a git repository"
            )

        monkeypatch.setattr(workspace_repo.subprocess, "run", _fake_run)
        assert workspace_repo.summarize_local_git_state(tmp_path) is None
        assert calls == [["git", "-C", str(tmp_path), "rev-parse", "HEAD"]]

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
