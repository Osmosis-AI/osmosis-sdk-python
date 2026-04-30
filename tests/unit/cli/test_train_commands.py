"""Tests for osmosis_ai.cli.commands.train."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import DetailResult, ListResult, OperationResult
from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.platform.api.models import (
    DeleteTrainingRunResult,
    PaginatedTrainingRuns,
    SubmitTrainingRunResult,
    TrainingRun,
    TrainingRunDetail,
)


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Swap console in both train_module and utils_module, return the output buffer.

    Sets a wide fixed width so Rich panels render their content on a
    single line per logical line, which lets tests assert on the
    rendered panel body with plain substring checks.
    """
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    monkeypatch.setattr(train_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


@pytest.fixture(autouse=True)
def _mock_workspace_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default `train submit` to skip the workspace<->git-repo check.

    Individual tests can re-mock this when exercising the validator.
    """
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.validate_active_workspace_repo",
        lambda **kwargs: None,
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
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Training Runs"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False

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
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.items[0]["name"] == "my-run"
        assert result.items[0]["model_name"] == "gpt-2"
        assert result.items[0]["eval_accuracy"] == 0.95

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
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.items[0]["name"] is None
        assert result.items[0]["status"] == "running"

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
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert len(result.items) == 1
        assert result.total_count == 5
        assert result.has_more is True


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
        result = train_module.status(name="run-1")

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run"
        assert result.data["name"] == "run-1"
        assert result.data["status"] == "completed"

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
        result = train_module.status(name="full-run")

        assert isinstance(result, DetailResult)
        fields = {field.label: field.value for field in result.fields}
        assert fields["Examples"] == "100"
        assert fields["Notes"] == "experiment notes"
        assert fields["HF Status"] == "uploaded"
        assert fields["Started"] == "2026-01-01"
        assert fields["Completed"] == "2026-01-02"
        assert result.data["examples_processed_count"] == 100
        assert result.data["notes"] == "experiment notes"
        assert result.data["hf_status"] == "uploaded"


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
    def _write_project(tmp_path: Path, *, rollout: str = "calculator") -> Path:
        for rel_path in (
            ".osmosis/research",
            f"rollouts/{rollout}",
            "configs/training",
            "configs/eval",
            "data",
        ):
            (tmp_path / rel_path).mkdir(parents=True, exist_ok=True)

        (tmp_path / ".osmosis" / "project.toml").write_text(
            "[project]\nsetup_source = 'test'\n",
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
        project_root = cls._write_project(tmp_path)
        path = project_root / "configs" / "training" / "train.toml"
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
        command_result = train_module.submit(config_path=config_path, yes=True)

        assert isinstance(command_result, OperationResult)
        assert command_result.operation == "train.submit"
        assert command_result.status == "success"
        assert command_result.resource is not None
        assert command_result.resource["name"] == "my-training-run"
        assert command_result.resource["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert command_result.resource["status"] == "pending"
        assert "/training/" in command_result.resource["url"]
        assert command_result.message == "Training run submitted: my-training-run"

    def test_submit_url_does_not_insert_rich_line_breaks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        result = self.SUBMIT_RESULT

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return result

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        command_result = train_module.submit(config_path=config_path, yes=True)

        expected_url = utils_module.platform_entity_url(
            "ws-test",
            "training",
            result.id,
        )
        assert isinstance(command_result, OperationResult)
        assert command_result.resource is not None
        assert command_result.resource["url"] == expected_url
        assert f"View: {expected_url}" in command_result.display_next_steps

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
        result = train_module.submit(config_path=config_path, yes=True)
        out = console_capture.getvalue()
        assert "calculator" in out
        assert "main.py" in out
        assert "Qwen/Qwen3.5-35B-A3B" in out
        assert "abc-123" in out
        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["config"] == {
            "rollout": "calculator",
            "entrypoint": "main.py",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "dataset": "abc-123",
            "commit_sha": None,
        }

    def test_submit_with_commit_sha(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        project_root = self._write_project(tmp_path, rollout="r")
        path = project_root / "configs" / "training" / "train.toml"
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
        result = train_module.submit(config_path=path, yes=True)
        assert captured_kwargs["commit_sha"] == "deadbeef"
        assert "deadbeef" in console_capture.getvalue()
        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["config"]["commit_sha"] == "deadbeef"

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
        self._write_project(tmp_path)
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
        project_root = self._write_project(tmp_path, rollout="graderless")
        (project_root / "rollouts" / "graderless" / "main.py").write_text(
            """
from osmosis_ai.rollout import AgentWorkflow


class TestWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None
""".strip(),
            encoding="utf-8",
        )
        path = project_root / "configs" / "training" / "graderless.toml"
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

    def test_submit_prints_remote_fetch_notice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        out = console_capture.getvalue()
        # Reminder always renders; "Before you submit" is the calm
        # variant when no warnings are detected.
        assert "Before you submit" in out
        assert "connected Git repository" in out
        assert "committed and pushed" in out

    def test_submit_warns_about_unpushed_commits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = self._write_config(tmp_path)

        monkeypatch.setattr(
            workspace_repo,
            "summarize_local_git_state",
            lambda _root: workspace_repo.LocalGitState(
                branch="main",
                head_sha="abcdef1234567890" + "0" * 24,
                is_dirty=False,
                has_upstream=True,
                ahead=2,
            ),
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        out = console_capture.getvalue()
        assert "Push before submitting" in out
        assert "2 unpushed commits" in out

    def test_submit_warns_about_uncommitted_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = self._write_config(tmp_path)

        monkeypatch.setattr(
            workspace_repo,
            "summarize_local_git_state",
            lambda _root: workspace_repo.LocalGitState(
                branch="feature",
                head_sha="0" * 40,
                is_dirty=True,
                has_upstream=True,
                ahead=0,
            ),
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        out = console_capture.getvalue()
        assert "Push before submitting" in out
        assert "Uncommitted changes" in out

    def test_submit_warns_when_branch_has_no_upstream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = self._write_config(tmp_path)

        monkeypatch.setattr(
            workspace_repo,
            "summarize_local_git_state",
            lambda _root: workspace_repo.LocalGitState(
                branch="local-only",
                head_sha="0" * 40,
                is_dirty=False,
                has_upstream=False,
                ahead=0,
            ),
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        out = console_capture.getvalue()
        assert "Push before submitting" in out
        assert "no upstream" in out
        assert "local-only" in out

    def test_submit_remote_fetch_notice_for_pinned_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        project_root = self._write_project(tmp_path, rollout="r")
        path = project_root / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
commit_sha = "deadbeef1234"
""".strip(),
            encoding="utf-8",
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=path, yes=True)

        out = console_capture.getvalue()
        assert "deadbeef1234" in out
        assert "already pushed to the remote" in out

    def test_submit_passes_workspace_and_project_to_validator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        captured: dict = {}

        def _capture(**kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(
            "osmosis_ai.platform.cli.workspace_repo.validate_active_workspace_repo",
            _capture,
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        assert captured["workspace_name"] == "ws-test"
        assert captured["project_root"] == config_path.parent.parent.parent
        assert captured["command_label"] == "`osmosis train submit`"

    def test_submit_propagates_workspace_repo_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)

        def _fail(**kwargs: object) -> None:
            raise CLIError(
                "`osmosis train submit` must be run from a clone of the workspace's connected Git repository."
            )

        monkeypatch.setattr(
            "osmosis_ai.platform.cli.workspace_repo.validate_active_workspace_repo",
            _fail,
        )

        with pytest.raises(CLIError, match="connected Git repository"):
            train_module.submit(config_path=config_path, yes=True)


class TestSubmitNonInteractiveContext:
    """`train submit` without --yes should surface its prompt context.

    AI agents and CI scripts can't see the Rich confirmation panel, so
    the JSON/plain error must carry the summary, notes, and warnings.
    """

    def test_submit_json_without_yes_includes_prompt_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        config_path = TestSubmit._write_config(tmp_path)

        class FakeClient:
            def submit_training_run(self, **kwargs):
                raise AssertionError("API must not be reached without confirmation")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        from osmosis_ai.cli import main as cli

        exit_code = cli.main(["--json", "train", "submit", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code != 0
        assert captured.out == ""
        envelope = json.loads(captured.err)
        assert envelope["error"]["code"] == "INTERACTIVE_REQUIRED"
        details = envelope["error"]["details"]
        assert details["prompt"] == "Submit this training run?"
        assert details["summary"] == {
            "Rollout": "calculator",
            "Entrypoint": "main.py",
            "Model": "Qwen/Qwen3.5-35B-A3B",
            "Dataset": "abc-123",
        }
        assert any("connected Git repository" in note for note in details["notes"])

    def test_submit_json_includes_git_warnings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = TestSubmit._write_config(tmp_path)

        monkeypatch.setattr(
            workspace_repo,
            "summarize_local_git_state",
            lambda _root: workspace_repo.LocalGitState(
                branch="feature",
                head_sha="0" * 40,
                is_dirty=True,
                has_upstream=True,
                ahead=3,
            ),
        )

        from osmosis_ai.cli import main as cli

        exit_code = cli.main(["--json", "train", "submit", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code != 0
        envelope = json.loads(captured.err)
        warnings = envelope["error"]["details"]["warnings"]
        assert any("Uncommitted changes" in w for w in warnings)
        assert any("3 unpushed commits" in w for w in warnings)

    def test_submit_plain_without_yes_writes_context_to_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        config_path = TestSubmit._write_config(tmp_path)

        class FakeClient:
            def submit_training_run(self, **kwargs):
                raise AssertionError("API must not be reached without confirmation")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        from osmosis_ai.cli import main as cli

        exit_code = cli.main(["--plain", "train", "submit", str(config_path)])
        captured = capsys.readouterr()

        assert exit_code != 0
        # plain stderr carries the prompt + summary + at least one note
        assert "Confirmation required: Submit this training run?" in captured.err
        assert "Rollout: calculator" in captured.err
        assert "Dataset: abc-123" in captured.err
        assert "Notes:" in captured.err
        # final error line is still surfaced for humans
        assert "Use --yes to confirm in non-interactive mode." in captured.err

    def test_require_confirmation_includes_details_in_cli_error(
        self,
    ) -> None:
        with override_output_context(format=OutputFormat.plain, interactive=False):
            with pytest.raises(CLIError) as exc_info:
                train_module._require_confirmation(
                    "Submit this training run?",
                    yes=False,
                    summary=[("Model", "Qwen")],
                    notes=["fetched from git"],
                    warnings=["unpushed commits"],
                )

        err = exc_info.value
        assert err.code == "INTERACTIVE_REQUIRED"
        assert err.details["prompt"] == "Submit this training run?"
        assert err.details["summary"] == {"Model": "Qwen"}
        assert err.details["notes"] == ["fetched from git"]
        assert err.details["warnings"] == ["unpushed commits"]


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
        result = train_module.delete(name="my-run", yes=True)

        assert isinstance(result, OperationResult)
        assert result.operation == "train.delete"
        assert result.status == "success"
        assert result.resource == {"name": "my-run"}
        assert result.message == 'Training run "my-run" deleted.'
