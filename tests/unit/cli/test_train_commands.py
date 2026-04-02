"""Tests for osmosis_ai.cli.commands.train."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    DeleteTrainingRunResult,
    PaginatedTrainingRuns,
    PreservedModel,
    TrainingRun,
    TrainingRunDetail,
)


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Swap console in both train_module and utils_module, return the output buffer."""
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(train_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project._require_auth",
        lambda: ("ws-test", object()),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project._resolve_project_id",
        lambda _p, *, workspace_name: "proj_1",
    )


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_training_runs(self, pid, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(project=None, limit=30, all_=False)
        assert "No training runs found" in console_capture.getvalue()

    def test_list_with_runs(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name="my-run",
            status="completed",
            model_name="gpt-2",
            eval_accuracy=0.95,
            created_at="2026-01-01T00:00:00Z",
        )

        class FakeClient:
            def list_training_runs(self, pid, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(project=None, limit=30, all_=False)
        out = console_capture.getvalue()
        assert "my-run" in out
        assert "abcdef12" in out
        assert "gpt-2" in out
        assert "acc:0.95" in out

    def test_list_unnamed_run(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name=None,
            status="running",
        )

        class FakeClient:
            def list_training_runs(self, pid, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(project=None, limit=30, all_=False)
        assert "(unnamed)" in console_capture.getvalue()

    def test_list_has_more(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name="r1",
            status="completed",
        )

        class FakeClient:
            def list_training_runs(self, pid, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=5, has_more=True
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(project=None, limit=30, all_=False)
        out = console_capture.getvalue()
        assert "Showing 1 of 5 training runs" in out
        assert "--all" in out


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_basic(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="completed",
            model_name="gpt-2",
        )

        class FakeClient:
            def list_training_runs(self, pid, *, limit=50, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

            def get_training_run(self, run_id, *, credentials=None):
                return detail

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.status(id="abcdef1234567890abcdef1234567890", project=None)
        out = console_capture.getvalue()
        assert "run-1" in out
        assert "completed" in out

    def test_status_with_all_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="full-run",
            status="completed",
            model_name="gpt-2",
            output_model_id="model_out_1",
            examples_processed_count=100,
            notes="experiment notes",
            hf_status="uploaded",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-02T00:00:00Z",
            eval_accuracy=0.88,
            reward_increase_delta=0.12,
            error_message=None,
            creator_name="alice",
            created_at="2025-12-31T00:00:00Z",
        )

        class FakeClient:
            def list_training_runs(self, pid, *, limit=50, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

            def get_training_run(self, run_id, *, credentials=None):
                return detail

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.status(id="abcdef1234567890abcdef1234567890", project=None)
        out = console_capture.getvalue()
        assert "model_out_1" in out
        assert "100" in out
        assert "experiment notes" in out
        assert "uploaded" in out
        assert "2026-01-01" in out
        assert "2026-01-02" in out


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    RUN_ID = "abcdef1234567890abcdef1234567890"

    def test_delete_without_output_model(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_training_runs(self, pid, *, limit=50, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

            def delete_training_run(self, run_id, *, credentials=None):
                return DeleteTrainingRunResult(deleted=True)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.delete(id=self.RUN_ID, project=None, yes=True)
        out = console_capture.getvalue()
        assert "deleted" in out
        assert "Output model preserved" not in out

    def test_delete_with_preserved_output_model(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_training_runs(self, pid, *, limit=50, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

            def delete_training_run(self, run_id, *, credentials=None):
                return DeleteTrainingRunResult(
                    deleted=True,
                    preserved_output_model=PreservedModel(
                        id="model123456789012345678901234",
                        name="my-output-model",
                    ),
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.delete(id=self.RUN_ID, project=None, yes=True)
        out = console_capture.getvalue()
        assert "deleted" in out
        assert "Output model preserved" in out
        assert "my-output-model" in out
