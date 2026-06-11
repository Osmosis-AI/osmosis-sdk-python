"""Tests for osmosis_ai.cli.commands.eval."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.commands.eval as eval_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.eval as platform_eval_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.output import DetailResult, ListResult
from osmosis_ai.platform.api.models import (
    EvaluationRun,
    EvaluationRunDetail,
    LogEntry,
    LogsPage,
    PaginatedEvaluationRuns,
)
from osmosis_ai.platform.auth import PlatformAPIError

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
FAKE_CREDENTIALS = object()


def _fake_eval_metrics(self, eval_run_id, *, git_identity, credentials=None):
    raise PlatformAPIError("not available")


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    monkeypatch.setattr(platform_eval_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def _git_context():
        return SimpleNamespace(
            workspace_directory="/repo",
            git_identity=GIT_IDENTITY,
            repo_url=REPO_URL,
            credentials=FAKE_CREDENTIALS,
        )

    monkeypatch.setattr(
        platform_eval_module,
        "require_git_workspace_directory_context",
        _git_context,
    )


class TestListEvalRuns:
    def test_list_display_columns_and_json_items_include_result_summary(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        results = {
            "score": 0.8123,
            "total_runs": 90,
            "sampled_rows": 100,
            "graded": 80,
            "passed": 60,
            "failed": 20,
            "skipped": 10,
            "pass_rate": 0.75,
        }
        run = EvaluationRun(
            id="eval-1",
            name="math-eval",
            status="stopped",
            created_at="2026-01-01T00:00:00Z",
            model={"name": "openai/gpt-5-mini"},
            dataset={"id": "dataset-1", "name": "eval.jsonl"},
            rollout={"id": "rollout-1", "name": "math-rollout"},
            creator_name="alice",
            results=results,
            config={"n": 2},
        )

        class FakeClient:
            def list_eval_runs(
                self,
                limit=30,
                offset=0,
                *,
                git_identity,
                credentials=None,
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedEvaluationRuns(
                    eval_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        result = eval_module.eval_list(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert [column.label for column in result.columns] == [
            "Name",
            "Status",
            "Rollout",
            "Reward",
            "Submitted",
            "Submitted By",
        ]
        assert [column.key for column in result.columns] == [
            "name",
            "status",
            "rollout",
            "avg_reward",
            "created_at",
            "creator_name",
        ]
        assert result.display_items is not None
        item = result.display_items[0]
        assert item["dataset"] == "eval.jsonl"
        assert item["model"] == "openai/gpt-5-mini"
        assert item["rollout"] == "math-rollout"
        assert item["avg_reward"] == "0.81"
        assert item["status"] == "[dim]\\[stopped][/dim]"

        json_item = result.items[0]
        assert json_item["results"] == results
        assert json_item["summary"] == {
            "avg_reward": 0.8123,
            "pass_rate": 0.75,
            "graded": 80,
            "passed": 60,
            "failed": 20,
            "skipped": 10,
            "progress": {
                "completed": 90,
                "total": 200,
                "unit": "samples",
            },
        }


class TestEvalInfo:
    def test_info_displays_submitted_richer_results_and_debug_context(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = EvaluationRunDetail(
            id="eval-1",
            name="math-eval",
            status="running",
            created_at="2026-01-01T00:00:00Z",
            started_at="2026-01-01T00:01:00Z",
            completed_at="2026-01-01T00:01:12.500Z",
            model={"name": "openai/gpt-5-mini"},
            dataset={"id": "dataset-1", "name": "eval.jsonl"},
            rollout={"id": "rollout-1", "name": "math-rollout"},
            creator_name="alice",
            config={"n": 2, "batch_size": 5, "pass_threshold": 0.5},
            entrypoint="main.py",
            commit_sha="abcdef1234567890",
            env_config={"PROMPT_MODE": "strict"},
            resolved_secret_scopes={
                "OPENAI_API_KEY": "workspace",
                "ANTHROPIC_API_KEY": "user_override",
            },
            results={
                "total_runs": 5,
                "sampled_rows": 4,
                "graded": 4,
                "passed": 3,
                "failed": 1,
                "skipped": 0,
                "score": 0.8123,
                "pass_rate": 0.75,
                "pass_threshold": 0.5,
                "total_tokens": 12345,
                "total_dataset_rows": 1000,
                "reward_stats": {
                    "min": 0.1,
                    "median": 0.8,
                    "max": 1.0,
                    "std": 0.2,
                    "pass_at_k": {"1": 0.5, "2": 0.75},
                },
            },
        )

        class FakeClient:
            def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
                assert name_or_id == "math-eval"
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return detail

            get_eval_run_metrics = _fake_eval_metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        result = eval_module.eval_info("math-eval", output=None)

        assert isinstance(result, DetailResult)
        field_rows = [(field.label, field.value) for field in result.fields]
        # Main info mirrors the Platform detail page sidebar.
        assert [label for label, _value in field_rows] == [
            "Name",
            "Status",
            "Progress",
            "Duration",
            "Pass Rate",
            "Tokens Used",
            "Submitted",
            "Submitted By",
            "Started",
            "Completed",
            "Dataset",
            "Model",
            "Rollout",
        ]
        fields = dict(field_rows)
        assert fields["Progress"] == "5 / 8 samples"
        assert fields["Duration"] == "12.5s"
        assert fields["Pass Rate"] == "75.0%"
        assert fields["Tokens Used"] == "12,345"
        assert fields["Submitted By"] == "alice"

        # Configuration + results live in their own sections, not the main info.
        section_plain = {
            line.split(": ", 1)[0]: line.split(": ", 1)[1]
            for section in result.sections
            for line in section.plain_lines
            if ": " in line
        }
        section_titles = [
            section.plain_lines[0].rstrip(":")
            for section in result.sections
            if section.plain_lines
        ]
        assert section_titles == ["Configuration", "Results"]
        assert section_plain["Entrypoint"] == "main.py"
        assert section_plain["Config"] == "n=2, batch_size=5, pass_threshold=0.5"
        assert section_plain["Commit"] == "abcdef1"
        assert section_plain["Secrets"] == (
            "ANTHROPIC_API_KEY (personal, overrides workspace), "
            "OPENAI_API_KEY (workspace)"
        )
        assert section_plain["Environment Variables"] == "PROMPT_MODE=strict"
        assert section_plain["Results"] == "4 graded, 3 passed, 1 failed, 0 skipped"
        assert section_plain["Pass Threshold"] == "0.5000"
        assert (
            section_plain["Reward Stats"]
            == "mean 0.8123, std 0.2000, min 0.1000, median 0.8000, max 1.0000"
        )
        assert section_plain["Pass@k"] == "1: 50.0%, 2: 75.0%"

        # Errors are dropped from the rendered output but kept in JSON data.
        assert "Error" not in fields
        assert "Error" not in section_plain
        # The detail endpoint stopped embedding logs; `osmosis eval logs` is
        # the replacement.
        assert "recent_logs" not in result.data
        assert result.data["summary"] == {
            "avg_reward": 0.8123,
            "pass_rate": 0.75,
            "graded": 4,
            "passed": 3,
            "failed": 1,
            "skipped": 0,
            "pass_threshold": 0.5,
            "total_tokens": 12345,
            "dataset_rows": 1000,
            "progress": {
                "completed": 5,
                "total": 8,
                "unit": "samples",
            },
        }
        assert "Creator" not in fields
        assert "Created" not in fields
        assert "Score" not in fields
        assert "Note" not in fields

    def test_info_shows_id_for_internal_user(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = EvaluationRunDetail(
            id="eval-1",
            name="math-eval",
            status="running",
            created_at="2026-01-01T00:00:00Z",
            is_internal_user=True,
        )

        class FakeClient:
            def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
                return detail

            get_eval_run_metrics = _fake_eval_metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        result = eval_module.eval_info("math-eval", output=None)

        assert isinstance(result, DetailResult)
        labels = [field.label for field in result.fields]
        assert labels[:3] == ["Name", "ID", "Status"]

    def test_info_progress_and_summary_agree_when_runs_exceed_total(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        # ``total_runs`` > ``sampled_rows * n`` (e.g. resumed/retried runs):
        # the rendered Progress string and the JSON summary must derive from the
        # same clamped numbers and never show ``N / M`` with ``N > M``.
        detail = EvaluationRunDetail(
            id="eval-2",
            name="overflow-eval",
            status="running",
            created_at="2026-01-01T00:00:00Z",
            config={"n": 2},
            results={"total_runs": 12, "sampled_rows": 4},
        )

        class FakeClient:
            def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
                return detail

            get_eval_run_metrics = _fake_eval_metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        result = eval_module.eval_info("overflow-eval", output=None)

        assert isinstance(result, DetailResult)
        fields = {field.label: field.value for field in result.fields}
        # total widened to completed (total is only a lower bound here).
        assert fields["Progress"] == "12 / 12 samples"
        assert result.data["summary"]["progress"] == {
            "completed": 12,
            "total": 12,
            "unit": "samples",
        }

    def test_info_failed_run_suggests_logs_command(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = EvaluationRunDetail(
            id="eval-3",
            name="broken-eval",
            status="failed",
            created_at="2026-01-01T00:00:00Z",
        )

        class FakeClient:
            def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
                return detail

            get_eval_run_metrics = _fake_eval_metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        result = eval_module.eval_info("broken-eval", output=None)

        assert "See logs with: osmosis eval logs broken-eval" in result.display_hints


class TestEvalLogs:
    LOG_ENTRIES = [
        LogEntry(
            timestamp="2026-06-01T00:00:00Z",
            level="info",
            step="init",
            message="Run created",
        ),
        LogEntry(
            timestamp="2026-06-01T00:05:00Z",
            level="error",
            step="eval",
            message="Missing API key",
            details={"secret": "OPENAI_API_KEY"},
        ),
    ]

    @staticmethod
    def _install_client(
        monkeypatch: pytest.MonkeyPatch, page: LogsPage
    ) -> dict[str, object]:
        captured: dict[str, object] = {}

        class FakeClient:
            def get_eval_run_logs(
                self, name_or_id, *, limit, cursor=None, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                captured["name_or_id"] = name_or_id
                captured["limit"] = limit
                captured["cursor"] = cursor
                return page

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)
        return captured

    def test_logs_renders_chronological_table(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = self._install_client(
            monkeypatch, LogsPage(logs=self.LOG_ENTRIES, next_cursor=None)
        )

        result = eval_module.eval_logs(name_or_id="eval-1", limit=50, cursor=None)

        assert captured == {"name_or_id": "eval-1", "limit": 50, "cursor": None}
        assert isinstance(result, ListResult)
        assert result.title == "Evaluation Run Logs: eval-1"
        assert [column.label for column in result.columns] == [
            "Time",
            "Level",
            "Step",
            "Message",
        ]
        # Oldest-first order is preserved from the server page.
        assert [item["message"] for item in result.items] == [
            "Run created",
            "Missing API key",
        ]
        assert result.items[0]["timestamp"] == "2026-06-01T00:00:00Z"
        assert result.items[1]["details"] == {"secret": "OPENAI_API_KEY"}
        assert result.total_count == 2
        assert result.has_more is False
        assert result.next_offset is None
        assert result.extra == {
            "next_cursor": None,
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
        }
        assert result.display_items is not None
        # Display timestamps are localized, raw ISO stays in items for JSON.
        assert result.display_items[0]["timestamp"] != "2026-06-01T00:00:00Z"
        assert result.display_items[0]["timestamp"].startswith("2026-")
        assert result.display_hints == ["Use osmosis eval info eval-1 for run details."]

    def test_logs_has_more_when_next_cursor_present(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        self._install_client(
            monkeypatch,
            LogsPage(logs=self.LOG_ENTRIES, next_cursor="2026-06-01T00:00:00Z|log-1"),
        )

        result = eval_module.eval_logs(name_or_id="eval-1", limit=2)

        assert isinstance(result, ListResult)
        assert result.has_more is True
        assert result.extra["next_cursor"] == "2026-06-01T00:00:00Z|log-1"

    def test_logs_passes_cursor_to_client(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = self._install_client(
            monkeypatch, LogsPage(logs=self.LOG_ENTRIES, next_cursor=None)
        )

        eval_module.eval_logs(
            name_or_id="eval-1", limit=50, cursor="2026-06-01T00:00:00Z|log-1"
        )

        assert captured["cursor"] == "2026-06-01T00:00:00Z|log-1"

    def test_logs_empty_page(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        self._install_client(monkeypatch, LogsPage(logs=[], next_cursor=None))

        result = eval_module.eval_logs(name_or_id="eval-1", limit=50)

        assert isinstance(result, ListResult)
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False

    @pytest.mark.parametrize("limit", ["0", "201"])
    def test_logs_rejects_out_of_range_limit(self, limit: str, capsys) -> None:
        from osmosis_ai.cli.main import main

        rc = main(["eval", "logs", "eval-1", "--limit", limit])

        assert rc == 2
        assert "is not in the range 1<=x<=200" in capsys.readouterr().err

    def test_logs_unknown_run_emits_not_found_envelope(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class FakeClient:
            def get_eval_run_logs(
                self, name_or_id, *, limit, cursor=None, git_identity, credentials=None
            ):
                raise PlatformAPIError("Evaluation run not found", 404)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_eval_module, "OsmosisClient", FakeClient)

        from osmosis_ai.cli import main as cli

        exit_code = cli.main(["--json", "eval", "logs", "missing-run"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        envelope = json.loads(captured.err)
        assert envelope["error"]["code"] == "NOT_FOUND"
        assert envelope["error"]["message"] == "Evaluation run not found"
        assert envelope["command"] == "eval logs"


class TestFormatPassAtK:
    def test_single_value_is_shown(self) -> None:
        results = {"reward_stats": {"pass_at_k": {"1": 0.5}}}
        assert platform_eval_module._format_pass_at_k(results) == "1: 50.0%"

    def test_no_values_returns_none(self) -> None:
        results = {"reward_stats": {"pass_at_k": {"1": "bad"}}}
        assert platform_eval_module._format_pass_at_k(results) is None
