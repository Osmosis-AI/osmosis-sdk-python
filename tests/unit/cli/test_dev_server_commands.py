"""Tests for osmosis_ai.platform.cli.dev_server (dev server up/down/list)."""

from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dev_server as dev_server_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import ListResult, OperationResult

GIT_IDENTITY = "acme/rollouts"
FAKE_CREDENTIALS = object()
FAKE_SERVER = {
    "id": "r1",
    "url": "https://r1.rollout-staging.gulp.dev",
    "expires_at": None,
}
FAKE_HEAD_SHA = "abc123def456abc123def456abc123def456abc1"


def _fake_ctx():
    return type(
        "FakeCtx",
        (),
        {"credentials": FAKE_CREDENTIALS, "git_identity": GIT_IDENTITY},
    )()


def _fake_git_state(*, is_dirty=False, head_sha=FAKE_HEAD_SHA):
    return type(
        "FakeGitState",
        (),
        {"head_sha": head_sha, "is_dirty": is_dirty},
    )()


def _fake_pinned_check(*, error=None, warnings=()):
    """Return a fake PinnedCommitCheck-like object."""
    return type(
        "FakePinnedCommitCheck",
        (),
        {"error": error, "warnings": warnings},
    )()


class TestDevServerUp:
    def test_up_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        captured: dict = {}

        class FakeClient:
            def provision_dev_rollout_server(
                self,
                *,
                rollout_name,
                commit_sha,
                repository_path,
                entrypoint,
                ttl_hours,
                credentials=None,
                git_identity,
            ):
                captured["rollout_name"] = rollout_name
                captured["commit_sha"] = commit_sha
                captured["repository_path"] = repository_path
                captured["ttl_hours"] = ttl_hours
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return FAKE_SERVER

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.up(ttl_hours=24, yes=False)

        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["url"] == FAKE_SERVER["url"]
        assert captured["rollout_name"] == "multiply"
        assert captured["commit_sha"] == FAKE_HEAD_SHA
        assert captured["repository_path"] == "rollouts/multiply"
        assert captured["ttl_hours"] == 24

    def test_up_no_ttl_passes_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        captured: dict = {}

        class FakeClient:
            def provision_dev_rollout_server(self, *, ttl_hours, **kwargs):
                captured["ttl_hours"] = ttl_hours
                return FAKE_SERVER

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.up(ttl_hours=None, yes=False)

        assert isinstance(result, OperationResult)
        assert captured["ttl_hours"] is None

    def test_up_dirty_tree_with_yes_proceeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Dirty tree + --yes → no prompt, client is called."""
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        # require_confirmation with yes=True is a no-op, so no need to mock it
        client_called = {}

        class FakeClient:
            def provision_dev_rollout_server(self, **kwargs):
                client_called["called"] = True
                return FAKE_SERVER

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.up(ttl_hours=24, yes=True)

        assert isinstance(result, OperationResult)
        assert client_called.get("called") is True

    def test_up_dirty_tree_declined_aborts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Dirty tree + no --yes + confirmation declined → aborts (no client call)."""
        import typer

        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        # Mock require_confirmation to raise Exit(0) simulating decline
        def _mock_require_confirmation(*args, **kwargs):
            raise typer.Exit(0)

        monkeypatch.setattr(
            dev_server_module, "require_confirmation", _mock_require_confirmation
        )

        client_called = {}

        class FakeClient:
            def provision_dev_rollout_server(self, **kwargs):
                client_called["called"] = True
                return FAKE_SERVER

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        with pytest.raises(typer.Exit):
            dev_server_module.up(ttl_hours=24, yes=False)

        assert not client_called.get("called")

    def test_up_commit_not_pushed_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Commit not found on remote → CLIError (hard error)."""
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(
                error=f"Pinned commit {FAKE_HEAD_SHA} exists locally but was not found on origin"
            ),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        with pytest.raises(CLIError, match="not found on origin"):
            dev_server_module.up(ttl_hours=24, yes=False)

    def test_up_missing_main_py_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        with pytest.raises(CLIError, match=r"main\.py"):
            dev_server_module.up(ttl_hours=24, yes=False)

    def test_up_missing_main_py_takes_priority_over_dirty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing main.py errors before the dirty-tree confirmation is reached."""
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)  # no main.py

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(
            dev_server_module,
            "check_pinned_commit",
            lambda **kwargs: _fake_pinned_check(),
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("require_confirmation should not be reached")

        monkeypatch.setattr(dev_server_module, "require_confirmation", _fail_if_called)

        with pytest.raises(CLIError, match=r"main\.py"):
            dev_server_module.up(ttl_hours=24, yes=False)


class TestDevServerDown:
    def test_down_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        class FakeClient:
            def teardown_dev_rollout_server(
                self, server_id, *, credentials=None, git_identity
            ):
                assert server_id == "r1"
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return {"id": "r1", "status": "stopped"}

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.down("r1")

        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["status"] == "stopped"


class TestDevServerList:
    def test_list_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_servers() returns a ListResult with the server rows from the API."""
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        fake_servers = [
            {
                "rollout_id": "r1",
                "rollout_name": "multiply",
                "url": "https://r1.rollout-staging.gulp.dev",
                "expires_at": None,
                "status": "running",
            }
        ]

        class FakeClient:
            def list_dev_rollout_servers(self, *, credentials=None, git_identity):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return {"servers": fake_servers}

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers()

        assert isinstance(result, ListResult)
        assert result.total_count == 1
        assert len(result.items) == 1
        assert result.items[0]["rollout_id"] == "r1"
        assert result.items[0]["rollout_name"] == "multiply"
        assert result.items[0]["status"] == "running"

    def test_list_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_servers() returns a ListResult with zero items when API returns empty."""
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        class FakeClient:
            def list_dev_rollout_servers(self, *, credentials=None, git_identity):
                return {"servers": []}

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers()

        assert isinstance(result, ListResult)
        assert result.total_count == 0
        assert result.items == []
