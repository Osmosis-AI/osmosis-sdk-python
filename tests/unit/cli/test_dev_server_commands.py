"""Tests for osmosis_ai.platform.cli.dev_server (dev server up/down)."""

from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dev_server as dev_server_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult

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

        result = dev_server_module.up(ttl_hours=24)

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
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        captured: dict = {}

        class FakeClient:
            def provision_dev_rollout_server(self, *, ttl_hours, **kwargs):
                captured["ttl_hours"] = ttl_hours
                return FAKE_SERVER

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.up(ttl_hours=None)

        assert isinstance(result, OperationResult)
        assert captured["ttl_hours"] is None

    def test_up_dirty_working_tree_raises(
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
            lambda cwd: _fake_git_state(is_dirty=True),
        )
        monkeypatch.setattr(
            dev_server_module, "git_worktree_top_level", lambda cwd: tmp_path
        )
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        with pytest.raises(CLIError, match="dirty"):
            dev_server_module.up(ttl_hours=24)

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
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: rollout_dir))

        with pytest.raises(CLIError, match=r"main\.py"):
            dev_server_module.up(ttl_hours=24)


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
