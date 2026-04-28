"""Tests for osmosis_ai.cli.commands.train."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.errors import CLIError
from osmosis_ai.platform.api.models import (
    DeleteTrainingRunResult,
    PaginatedTrainingRuns,
    SubmitTrainingRunResult,
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
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_training_runs(self, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(limit=30, all_=False)
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
            def list_training_runs(self, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(limit=30, all_=False)
        out = console_capture.getvalue()
        assert "my-run" in out
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
            def list_training_runs(self, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(limit=30, all_=False)
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
            def list_training_runs(self, limit=30, offset=0, *, credentials=None):
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=5, has_more=True
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.list_runs(limit=30, all_=False)
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
            def get_training_run(self, run_id, *, credentials=None):
                return detail

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.status(name="run-1")
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
            def get_training_run(self, run_id, *, credentials=None):
                return detail

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.status(name="full-run")
        out = console_capture.getvalue()
        assert "100" in out
        assert "experiment notes" in out
        assert "uploaded" in out
        assert "2026-01-01" in out
        assert "2026-01-02" in out


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    SUBMIT_RESULT = SubmitTrainingRunResult(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="my-training-run",
        status="pending",
        created_at="2026-04-10T12:00:00Z",
    )

    @staticmethod
    def _write_workspace(tmp_path: Path, *, rollout: str = "calculator") -> Path:
        for rel_path in (
            ".osmosis/research",
            f"rollouts/{rollout}",
            "configs/training",
            "configs/eval",
            "data",
        ):
            (tmp_path / rel_path).mkdir(parents=True, exist_ok=True)

        (tmp_path / ".osmosis" / "workspace.toml").write_text(
            "[workspace]\nsetup_source = 'test'\n",
            encoding="utf-8",
        )
        (tmp_path / "rollouts" / rollout / "main.py").write_text(
            """
from osmosis_ai.rollout import AgentWorkflow, Grader


class TestWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class TestGrader(Grader):
    async def grade(self, ctx):
        return 1.0
""".strip(),
            encoding="utf-8",
        )
        return tmp_path

    @classmethod
    def _write_config(cls, tmp_path: Path) -> Path:
        workspace_root = cls._write_workspace(tmp_path)
        path = workspace_root / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.5-35B-A3B"
dataset = "abc-123"

[training]
lr = 1e-6
total_epochs = 1
""".strip(),
            encoding="utf-8",
        )
        return path

    def test_submit_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        result = self.SUBMIT_RESULT

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return result

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)
        out = console_capture.getvalue()
        assert "my-training-run" in out
        assert "550e8400" in out
        assert "pending" in out
        assert "training" in out  # URL contains "training"

    def test_submit_shows_summary_table(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return self.SUBMIT_RESULT

        FakeClient.SUBMIT_RESULT = self.SUBMIT_RESULT
        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)
        out = console_capture.getvalue()
        assert "calculator" in out
        assert "main.py" in out
        assert "Qwen/Qwen3.5-35B-A3B" in out
        assert "abc-123" in out

    def test_submit_with_commit_sha(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        workspace_root = self._write_workspace(tmp_path, rollout="r")
        path = workspace_root / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
commit_sha = "deadbeef"
""".strip(),
            encoding="utf-8",
        )

        captured_kwargs: dict = {}

        class FakeClient:
            def submit_training_run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=path, yes=True)
        assert captured_kwargs["commit_sha"] == "deadbeef"
        assert "deadbeef" in console_capture.getvalue()

    def test_submit_passes_config_to_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        captured_kwargs: dict = {}

        class FakeClient:
            def submit_training_run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)
        assert captured_kwargs["model_path"] == "Qwen/Qwen3.5-35B-A3B"
        assert captured_kwargs["dataset"] == "abc-123"
        assert captured_kwargs["rollout_name"] == "calculator"
        assert captured_kwargs["entrypoint"] == "main.py"
        assert captured_kwargs["config"] == {"lr": 1e-6, "total_epochs": 1}

    def test_submit_rejects_non_canonical_training_config_path(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_workspace(tmp_path)
        path = tmp_path / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(CLIError, match="configs/training"):
            train_module.submit(config_path=path, yes=True)

    def test_submit_requires_concrete_grader(
        self,
        tmp_path: Path,
    ) -> None:
        workspace_root = self._write_workspace(tmp_path, rollout="graderless")
        (workspace_root / "rollouts" / "graderless" / "main.py").write_text(
            """
from osmosis_ai.rollout import AgentWorkflow


class TestWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None
""".strip(),
            encoding="utf-8",
        )
        path = workspace_root / "configs" / "training" / "graderless.toml"
        path.write_text(
            """
[experiment]
rollout = "graderless"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(CLIError, match="Grader"):
            train_module.submit(config_path=path, yes=True)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_basic(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def delete_training_run(self, run_id, *, credentials=None):
                return DeleteTrainingRunResult(deleted=True)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.delete(name="my-run", yes=True)
        out = console_capture.getvalue()
        assert "deleted" in out
        assert "my-run" in out
