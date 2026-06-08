"""Tests for osmosis_ai.cli.commands.eval."""

from __future__ import annotations

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
            recent_logs=[
                {
                    "step": "eval",
                    "level": "error",
                    "message": "Missing API key",
                    "timestamp": "2026-01-01T00:01:12Z",
                }
            ],
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
            "ID",
            "Status",
            "Progress",
            "Duration",
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
        assert section_plain["Required Secrets"] == (
            "ANTHROPIC_API_KEY (personal, overrides workspace), "
            "OPENAI_API_KEY (workspace)"
        )
        assert section_plain["Environment"] == "PROMPT_MODE=strict"
        assert section_plain["Results"] == "4 graded, 3 passed, 1 failed, 0 skipped"
        assert section_plain["Avg. Reward"] == "0.8123"
        assert section_plain["Pass Rate"] == "75.0%"
        assert section_plain["Pass Threshold"] == "0.5000"
        assert (
            section_plain["Reward Stats"]
            == "min 0.1000, median 0.8000, max 1.0000, std 0.2000"
        )
        assert section_plain["Pass@k"] == "1: 50.0%, 2: 75.0%"
        assert section_plain["Total Tokens"] == "12,345"
        assert section_plain["Dataset Rows"] == "1,000"

        # Errors are dropped from the rendered output but kept in JSON data.
        assert "Error" not in fields
        assert "Error" not in section_plain
        assert result.data["recent_logs"] == [
            {
                "step": "eval",
                "level": "error",
                "message": "Missing API key",
                "timestamp": "2026-01-01T00:01:12Z",
            }
        ]
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
