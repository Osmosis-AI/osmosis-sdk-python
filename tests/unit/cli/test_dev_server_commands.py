"""Tests for osmosis_ai.platform.cli.dev_server (dev server up/down/list)."""

from __future__ import annotations

from pathlib import Path

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.dev_server as dev_server_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import ListResult, OperationResult
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.platform.api.models import PaginatedDevRolloutServers
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

GIT_IDENTITY = "acme/rollouts"
FAKE_CREDENTIALS = object()
FAKE_SERVER = {
    "id": "r1",
    "url": "https://r1.rollout-staging.gulp.dev",
    "expires_at": None,
}
FAKE_HEAD_SHA = "abc123def456abc123def456abc123def456abc1"


def _fake_ctx(workspace_directory: Path | None = None):
    return type(
        "FakeCtx",
        (),
        {
            "credentials": FAKE_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
            "workspace_directory": workspace_directory,
        },
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
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
        assert result.message is not None
        assert "provisioning" in result.message
        assert FAKE_SERVER["url"] in result.message
        assert "osmosis dev server list" in result.message

    def test_up_no_ttl_passes_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rollout_dir = tmp_path / "rollouts" / "multiply"
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "main.py").write_text("# main", encoding="utf-8")

        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(),
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
            lambda: _fake_ctx(workspace_directory=tmp_path),
        )
        monkeypatch.setattr(
            dev_server_module,
            "summarize_local_git_state",
            lambda cwd: _fake_git_state(is_dirty=True),
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
                "id": "r1",
                "name": "multiply",
                "url": "https://r1.rollout-staging.gulp.dev",
                "expires_at": None,
                "started_at": None,
                "status": "running",
            }
        ]

        class FakeClient:
            def list_dev_rollout_servers(
                self, limit, offset, *, credentials=None, git_identity
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedDevRolloutServers.from_dict(
                    {
                        "dev_rollout_servers": fake_servers,
                        "total_count": 1,
                        "has_more": False,
                        "next_offset": None,
                    }
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers(limit=DEFAULT_PAGE_SIZE, all_=False)

        assert isinstance(result, ListResult)
        assert result.total_count == 1
        assert len(result.items) == 1
        assert result.items[0]["id"] == "r1"
        assert result.items[0]["name"] == "multiply"
        assert result.items[0]["status"] == "running"
        assert result.items[0]["expires_at"] is None
        assert result.display_items is not None
        assert result.display_items[0]["expires_at"] == "No expiration"

    def test_list_display_items_no_expiration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """display_items shows 'No expiration' for None expires_at; items keeps original."""
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        fake_servers = [
            {
                "id": "r2",
                "name": "my-rollout",
                "url": "https://r2.rollout-staging.gulp.dev",
                "expires_at": None,
                "started_at": None,
                "status": "running",
            }
        ]

        class FakeClient:
            def list_dev_rollout_servers(
                self, limit, offset, *, credentials=None, git_identity
            ):
                return PaginatedDevRolloutServers.from_dict(
                    {
                        "dev_rollout_servers": fake_servers,
                        "total_count": 1,
                        "has_more": False,
                        "next_offset": None,
                    }
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers(limit=DEFAULT_PAGE_SIZE, all_=False)

        assert isinstance(result, ListResult)
        assert result.items[0]["expires_at"] is None
        assert result.display_items is not None
        assert result.display_items[0]["expires_at"] == "No expiration"
        assert result.display_items[0]["id"] == "r2"

    def test_list_display_items_formats_non_empty_expires_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """display_items renders a real expires_at as a local date string."""
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        fake_servers = [
            {
                "id": "r3",
                "name": "timed-rollout",
                "url": "https://r3.rollout-staging.gulp.dev",
                "expires_at": "2026-06-25T12:00:00Z",
                "started_at": None,
                "status": "running",
            }
        ]

        class FakeClient:
            def list_dev_rollout_servers(
                self, limit, offset, *, credentials=None, git_identity
            ):
                return PaginatedDevRolloutServers.from_dict(
                    {
                        "dev_rollout_servers": fake_servers,
                        "total_count": 1,
                        "has_more": False,
                        "next_offset": None,
                    }
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers(limit=DEFAULT_PAGE_SIZE, all_=False)

        assert isinstance(result, ListResult)
        assert result.items[0]["expires_at"] == "2026-06-25T12:00:00Z"
        assert result.display_items is not None
        assert result.display_items[0]["expires_at"] == format_local_date(
            "2026-06-25T12:00:00Z"
        )

    def test_list_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_servers() returns a ListResult with zero items when API returns empty."""
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )

        class FakeClient:
            def list_dev_rollout_servers(
                self, limit, offset, *, credentials=None, git_identity
            ):
                return PaginatedDevRolloutServers.from_dict(
                    {
                        "dev_rollout_servers": [],
                        "total_count": 0,
                        "has_more": False,
                        "next_offset": None,
                    }
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", FakeClient)

        result = dev_server_module.list_servers(limit=DEFAULT_PAGE_SIZE, all_=False)

        assert isinstance(result, ListResult)
        assert result.total_count == 0
        assert result.items == []


class TestDevServerLogs:
    """Tests for osmosis_ai.platform.cli.dev_server.logs."""

    @staticmethod
    def _page(messages, next_cursor=None):
        from osmosis_ai.platform.api.models import LogsPage

        return LogsPage.from_dict(
            {
                "logs": [
                    {"timestamp": f"t{i}", "message": m} for i, m in enumerate(messages)
                ],
                "next_cursor": next_cursor,
            }
        )

    def _install(self, monkeypatch, fake_client):
        monkeypatch.setattr(
            dev_server_module,
            "resolve_git_workspace_directory_context",
            lambda: _fake_ctx(),
        )
        monkeypatch.setattr(api_client_module, "OsmosisClient", fake_client)
        monkeypatch.setattr(dev_server_module, "OsmosisClient", fake_client)

    def test_json_oneshot_emits_ndjson(self, monkeypatch, capsys) -> None:
        import json

        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        outer = self

        class FakeClient:
            def get_dev_rollout_server_logs(
                self,
                server_id,
                *,
                limit,
                cursor=None,
                direction="older",
                credentials=None,
                git_identity,
            ):
                assert direction == "older"
                assert limit == 100
                return outer._page(["a", "b"], next_cursor="c1")

        self._install(monkeypatch, FakeClient)

        with override_output_context(format=OutputFormat.json) as out:
            with pytest.raises(typer.Exit) as exc:
                dev_server_module.logs("srv-1", follow=None, tail=100)
            assert exc.value.exit_code == 0
            assert out.output_emitted is True

        captured = capsys.readouterr().out
        lines = [ln for ln in captured.splitlines() if ln.strip()]
        assert json.loads(lines[0]) == {"timestamp": "t0", "message": "a"}
        assert json.loads(lines[1]) == {"timestamp": "t1", "message": "b"}
        assert "schema_version" not in captured

    def test_plain_oneshot_emits_tab_lines(self, monkeypatch, capsys) -> None:
        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        outer = self

        class FakeClient:
            def get_dev_rollout_server_logs(
                self, server_id, *, direction="older", **kw
            ):
                assert direction == "older"
                return outer._page(["hello"], next_cursor=None)

        self._install(monkeypatch, FakeClient)

        with override_output_context(format=OutputFormat.plain):
            with pytest.raises(typer.Exit):
                dev_server_module.logs("srv-1", follow=None, tail=100)

        captured = capsys.readouterr().out
        assert captured.splitlines()[0] == "t0\thello"

    def test_oneshot_does_not_follow(self, monkeypatch) -> None:
        """json/plain with follow unset makes exactly one (older) fetch."""
        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        outer = self
        calls = []

        class FakeClient:
            def get_dev_rollout_server_logs(
                self, server_id, *, direction="older", cursor=None, **kw
            ):
                calls.append((direction, cursor))
                return outer._page(["a"], next_cursor="c1")

        self._install(monkeypatch, FakeClient)

        with override_output_context(format=OutputFormat.json):
            with pytest.raises(typer.Exit):
                dev_server_module.logs("srv-1", follow=None, tail=100)

        assert calls == [("older", None)]

    @staticmethod
    def _entries(messages):
        from osmosis_ai.platform.api.models import LogEntry

        return [
            LogEntry.from_dict({"timestamp": f"t{i}", "message": m})
            for i, m in enumerate(messages)
        ]

    def _stream_client(self, messages, *, raise_ki=False, calls=None):
        outer = self

        class FakeClient:
            def stream_dev_rollout_server_logs(
                self, server_id, *, tail, credentials=None, git_identity
            ):
                if calls is not None:
                    calls.append(("stream", tail))
                yield from outer._entries(messages)
                if raise_ki:
                    raise KeyboardInterrupt

            def get_dev_rollout_server_logs(self, *a, **kw):
                raise AssertionError("follow must not use the paged GET")

        return FakeClient

    def test_plain_follow_streams_lines(self, monkeypatch, capsys) -> None:
        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        self._install(monkeypatch, self._stream_client(["a1", "b1"]))

        with override_output_context(format=OutputFormat.plain):
            with pytest.raises(typer.Exit) as exc:
                dev_server_module.logs("srv-1", follow=True, tail=100)
            assert exc.value.exit_code == 0

        captured = capsys.readouterr().out
        messages = [ln.split("\t", 1)[1] for ln in captured.splitlines() if "\t" in ln]
        assert messages == ["a1", "b1"]

    def test_json_follow_streams_ndjson(self, monkeypatch, capsys) -> None:
        import json

        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        self._install(monkeypatch, self._stream_client(["a1"]))

        with override_output_context(format=OutputFormat.json):
            with pytest.raises(typer.Exit):
                dev_server_module.logs("srv-1", follow=True, tail=100)

        captured = capsys.readouterr().out
        lines = [ln for ln in captured.splitlines() if ln.strip()]
        assert json.loads(lines[0]) == {"timestamp": "t0", "message": "a1"}
        assert "schema_version" not in captured

    def test_rich_follows_by_default(self, monkeypatch) -> None:
        """rich with follow unset streams (uses the SSE stream, not the paged GET)."""
        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        calls = []
        self._install(monkeypatch, self._stream_client(["a1"], calls=calls))

        with override_output_context(format=OutputFormat.rich, interactive=False):
            with pytest.raises(typer.Exit):
                dev_server_module.logs("srv-1", follow=None, tail=100)

        assert calls == [("stream", 100)]

    def test_follow_keyboardinterrupt_detaches_cleanly(self, monkeypatch) -> None:
        import typer

        from osmosis_ai.cli.output.context import OutputFormat, override_output_context

        self._install(monkeypatch, self._stream_client(["a1"], raise_ki=True))

        with override_output_context(format=OutputFormat.plain):
            with pytest.raises(typer.Exit) as exc:
                dev_server_module.logs("srv-1", follow=True, tail=100)
            assert exc.value.exit_code == 0
