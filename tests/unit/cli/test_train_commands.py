"""Tests for osmosis_ai.cli.commands.train."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path

import pytest

import osmosis_ai.cli.commands.train as train_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.shared_submit as shared_submit_module
import osmosis_ai.platform.cli.train as platform_train_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import DetailResult, ListResult, OperationResult
from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.platform.api.models import (
    LogEntry,
    LogsPage,
    MetricDataPoint,
    MetricHistory,
    MetricSummary,
    PaginatedTrainingRuns,
    SubmitRunResult,
    TrainingRun,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)
from osmosis_ai.platform.auth import PlatformAPIError
from tests.unit.submit_preflight_helpers import (
    assert_submit_aborts_on_invalid_commit,
    assert_submit_surfaces_commit_warning,
    disable_commit_preflight,
)

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
FAKE_CREDENTIALS = object()


def _git_extra() -> dict[str, object]:
    return {
        "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
        "workspace_directory": "/repo",
    }


def assert_git_context(data: dict[str, object]) -> None:
    assert data == _git_extra()


def _raise_metrics_unavailable(self, run_id, *, git_identity, credentials=None):
    assert git_identity == GIT_IDENTITY
    raise PlatformAPIError("not available")


def _find_temp_workspace_directory(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / ".osmosis").is_dir()
            and (candidate / "configs" / "training").is_dir()
            and (candidate / "rollouts").is_dir()
        ):
            return candidate
    return None


def _make_workspace_directory(root: Path, *, rollout: str = "demo") -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis",
        f"rollouts/{rollout}",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)

    (root / "rollouts" / rollout / "main.py").write_text(
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
    return root


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Swap console in both train_module and utils_module, return the output buffer.

    Sets a wide fixed width so Rich panels render their content on a
    single line per logical line, which lets tests assert on the
    rendered panel body with plain substring checks.
    """
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    monkeypatch.setattr(platform_train_module, "console", console)
    monkeypatch.setattr(shared_submit_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_git_context(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    if request.node.name == "test_train_submit_requires_linked_project":
        return

    def _git_context():
        workspace_directory = _find_temp_workspace_directory(Path.cwd()) or Path(
            "/repo"
        )
        return type(
            "GitContext",
            (),
            {
                "workspace_directory": workspace_directory.resolve(),
                "git_identity": GIT_IDENTITY,
                "repo_url": REPO_URL,
                "credentials": FAKE_CREDENTIALS,
            },
        )()

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_workspace_directory_context",
        _git_context,
    )
    monkeypatch.setattr(
        platform_train_module,
        "require_git_workspace_directory_context",
        _git_context,
    )
    monkeypatch.setattr(
        shared_submit_module,
        "require_git_workspace_directory_context",
        _git_context,
    )


@pytest.fixture(autouse=True)
def _mock_workspace_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default `train submit` to treat temp Osmosis workspace directories as Git roots."""

    def _git_top_level(start: Path) -> Path | None:
        current = start.resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".osmosis").is_dir():
                return candidate
        return None

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.git_worktree_top_level",
        _git_top_level,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.require_git_top_level",
        lambda *args, **kwargs: None,
        raising=False,
    )
    # Disable the pinned-commit preflight by default so submit tests don't make
    # real git/GitHub calls; tests that exercise it patch this explicitly.
    disable_commit_preflight(monkeypatch)


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_training_runs(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Training Runs"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False
        assert_git_context(result.extra)

    def test_list_with_runs(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name="my-run",
            status="completed",
            model_name="gpt-2",
            created_at="2026-01-01T00:00:00Z",
        )

        class FakeClient:
            def list_training_runs(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.items[0]["name"] == "my-run"
        assert result.items[0]["model_name"] == "gpt-2"

    def test_list_display_columns_show_training_summary_fields(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name="long-human-readable-run-name",
            status="running",
            dataset_name="train.jsonl",
            model_name="gpt-2",
            rollout_name="math-rollout",
            created_at="2026-01-01T00:00:00Z",
            current_step=31,
            total_steps=25,
            reward=0.75,
        )

        class FakeClient:
            def list_training_runs(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.list_runs(limit=30, all_=False)

        assert [column.label for column in result.columns] == [
            "Name",
            "Status",
            "Base Model",
            "Rollout",
            "Reward",
            "Submitted",
            "Submitted By",
        ]
        assert result.columns[0].key == "name"
        assert result.columns[0].ratio == 4
        assert result.columns[0].overflow == "fold"
        assert result.columns[1].key == "status"
        assert result.columns[2].key == "model_name"
        assert result.columns[3].key == "rollout_name"
        assert result.columns[3].overflow == "fold"
        assert result.columns[4].key == "reward"
        assert result.columns[5].key == "created_at"
        assert result.columns[6].key == "creator_name"
        assert result.display_items is not None
        assert result.display_items[0]["dataset_name"] == "train.jsonl"
        assert result.display_items[0]["model_name"] == "gpt-2"
        assert result.display_items[0]["rollout_name"] == "math-rollout"
        assert result.display_items[0]["reward"] == "0.75"
        assert result.items[0]["current_step"] == 31
        assert result.items[0]["total_steps"] == 25
        assert result.items[0]["reward"] == 0.75
        assert result.items[0]["summary"] == {
            "reward": 0.75,
            "progress": {
                "completed": 25,
                "total": 25,
                "unit": "steps",
            },
        }
        assert result.display_hints == ["Use osmosis train info <name> for details."]

    def test_list_unnamed_run(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        run = TrainingRun(
            id="abcdef1234567890abcdef1234567890",
            name=None,
            status="running",
        )

        class FakeClient:
            def list_training_runs(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
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
            def list_training_runs(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=5, has_more=True
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.list_runs(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert len(result.items) == 1
        assert result.total_count == 5
        assert result.has_more is True


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_combines_status_checkpoints_and_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.api.models import LoraCheckpointInfo

        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="finished",
            model_name="gpt-2",
            dataset_name="train.jsonl",
            rollout_name="math-rollout",
            platform_url="https://platform.osmosis.ai/ws/training/abcdef1234567890abcdef1234567890",
            current_step=31,
            total_steps=25,
            reward=0.75,
        )
        checkpoint = LoraCheckpointInfo(
            id="ckpt_abcdef123456",
            checkpoint_name="run-1-step-100",
            checkpoint_step=100,
            status="uploaded",
            created_at="2026-01-01T01:00:00Z",
        )
        metric_data = TrainingRunMetrics(
            training_run_id=detail.id,
            status="finished",
            overview=TrainingRunMetricsOverview(
                duration_ms=1000,
                metric_summaries=[
                    MetricSummary(
                        key="rollout/raw_reward",
                        title="Training Reward",
                        initial=0.50,
                        latest=0.75,
                        delta=0.25,
                        min=0.45,
                        max=0.78,
                    ),
                ],
                examples_processed_count=10,
            ),
            metrics=[
                MetricHistory(
                    metric_key="rollout/raw_reward",
                    title="Reward",
                    data_points=[MetricDataPoint(step=1, value=0.75, timestamp=0)],
                )
            ],
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return type("CheckpointPage", (), {"checkpoints": [checkpoint]})()

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):
                assert run_id == detail.id
                assert git_identity == GIT_IDENTITY
                return metric_data

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-1", output=str(tmp_path / "metrics.json"))

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run Info"
        assert result.data["training_run"]["name"] == "run-1"
        field_rows = [(field.label, field.value) for field in result.fields]
        assert ("Progress", "25 / 25 steps") in field_rows
        assert ("Dataset", "train.jsonl") in field_rows
        assert ("Rollout", "math-rollout") in field_rows
        assert result.data["training_run"]["current_step"] == 31
        assert result.data["training_run"]["total_steps"] == 25
        assert result.data["training_run"]["reward"] == 0.75
        assert result.data["summary"] == {
            "reward": 0.75,
            "progress": {
                "completed": 25,
                "total": 25,
                "unit": "steps",
            },
        }
        assert result.data["checkpoints"][0]["checkpoint_name"] == "run-1-step-100"
        assert result.data["metrics_available"] is True
        assert result.data["metrics"]["summary"]["training_reward"]["latest"] == 0.75
        assert result.data["output_path"] == str(tmp_path / "metrics.json")
        assert any("Saved metrics" in hint for hint in result.display_hints)
        assert {
            key: result.data[key] for key in ("git", "workspace_directory")
        } == _git_extra()

    def test_info_pending_run_does_not_fail_when_metrics_are_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="pending-run",
            status="pending",
            model_name="gpt-2",
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):  # pragma: no cover
                raise AssertionError("pending info should not fetch metrics")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="pending-run")

        assert isinstance(result, DetailResult)
        assert result.data["training_run"]["status"] == "pending"
        assert result.data["metrics_available"] is False
        assert "not yet available" in (result.data["metrics_error"] or "")
        assert all(field.label != "Note" for field in result.fields)

    def test_info_internal_user_sees_id_row_before_status(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="finished",
            model_name="gpt-2",
            dataset_name="train.jsonl",
            rollout_name="math-rollout",
            current_step=25,
            total_steps=25,
            is_internal_user=True,
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                return type("CheckpointPage", (), {"checkpoints": []})()

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):
                raise PlatformAPIError("not available")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-1")

        assert isinstance(result, DetailResult)
        labels = [field.label for field in result.fields]
        assert labels[:3] == ["Name", "ID", "Status"]
        if "Progress" in labels:
            assert labels.index("Progress") == labels.index("Status") + 1

    def test_info_metrics_fallback_progress_inserted_after_status(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-metrics-fallback",
            status="running",
            model_name="gpt-2",
            is_internal_user=False,
        )

        metrics = TrainingRunMetrics(
            training_run_id=detail.id,
            status="running",
            overview=TrainingRunMetricsOverview(
                duration_ms=None,
                metric_summaries=[],
                examples_processed_count=None,
                total_steps=100,
                latest_step=42,
            ),
            metrics=[],
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                return type("CheckpointPage", (), {"checkpoints": []})()

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):
                return metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-metrics-fallback", output=None)

        assert isinstance(result, DetailResult)
        labels = [field.label for field in result.fields]
        assert "Progress" in labels
        assert labels.index("Progress") == labels.index("Status") + 1


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
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                raise PlatformAPIError("not available")

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                raise PlatformAPIError("not available")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-1")

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run Info"
        assert result.data["training_run"]["name"] == "run-1"
        assert result.data["training_run"]["status"] == "completed"
        assert {
            key: result.data[key] for key in ("git", "workspace_directory")
        } == _git_extra()
        assert "workspace" not in result.data

    def test_status_renders_checkpoints_as_sections(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.platform.api.models import LoraCheckpointInfo

        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="finished",
            model_name="gpt-2",
            created_at="2026-01-01T00:00:00Z",
            platform_url="https://platform.osmosis.ai/ws/training/abcdef1234567890abcdef1234567890",
        )
        checkpoint = LoraCheckpointInfo(
            id="ckpt_abcdef123456",
            checkpoint_name="run-1-step-100",
            checkpoint_step=100,
            status="uploaded",
            created_at="2026-01-01T01:00:00Z",
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return type("CheckpointPage", (), {"checkpoints": [checkpoint]})()

            get_training_run_metrics = _raise_metrics_unavailable

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-1")

        assert all(field.label != "Checkpoint" for field in result.fields)
        assert all(field.label != "Deploy" for field in result.fields)
        assert result.sections
        assert result.display_hints == [
            f"View: {detail.platform_url}",
            "Deploy with: osmosis model deploy <lora-model-name>",
        ]
        assert result.data["checkpoints"][0]["checkpoint_name"] == "run-1-step-100"

    def test_status_checkpoint_section_omits_created_column_and_escapes_names(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from rich.console import Console as RichConsole

        from osmosis_ai.platform.api.models import LoraCheckpointInfo

        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="finished",
            model_name="gpt-2",
            created_at="2026-01-01T00:00:00Z",
        )
        checkpoint = LoraCheckpointInfo(
            id="ckpt_abcdef123456",
            checkpoint_name="[red]danger[/red]",
            checkpoint_step=100,
            status="uploaded",
            created_at="2026-01-01T12:34:00Z",
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return type("CheckpointPage", (), {"checkpoints": [checkpoint]})()

            get_training_run_metrics = _raise_metrics_unavailable

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="run-1")

        assert result.sections
        section = result.sections[0]
        assert [column.header for column in section.rich.columns] == [
            "Checkpoint",
            "Step",
            "Status",
        ]
        output = StringIO()
        rich = RichConsole(file=output, force_terminal=False, no_color=True, width=200)
        rich.print(section.rich)
        rendered = output.getvalue()

        assert "[red]danger[/red]" in rendered
        assert "Created" not in rendered
        assert section.plain_lines
        plain_line = section.plain_lines[0]
        assert "2026-01-01" not in plain_line
        assert ":00 " not in plain_line

    @pytest.mark.parametrize("status", ["failed", "crashed"])
    def test_status_failed_run_suggests_logs_command(
        self,
        status: str,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="broken-run",
            status=status,
            model_name="gpt-2",
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                raise PlatformAPIError("not available")

            get_training_run_metrics = _raise_metrics_unavailable

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="broken-run")

        assert "See logs with: osmosis train logs broken-run" in result.display_hints

    def test_status_uses_detailed_local_timestamps(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="timed-run",
            status="running",
            model_name="gpt-2",
            created_at="2026-01-01T00:00:00Z",
            started_at="2026-01-01T00:01:02Z",
            completed_at="2026-01-01T00:02:03Z",
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            get_training_run_metrics = _raise_metrics_unavailable

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="timed-run")
        fields = {field.label: field.value for field in result.fields}

        assert len(fields["Submitted"]) >= len("2026-01-01 00:00:00")
        assert len(fields["Started"]) >= len("2026-01-01 00:01:02")
        assert len(fields["Completed"]) >= len("2026-01-01 00:02:03")

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
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-02T00:00:00Z",
            creator_name="alice",
            created_at="2025-12-31T00:00:00Z",
            config={
                "model_path": "gpt-2",
                "training": {"lr": 0.001, "total_epochs": 3},
                "sampling": {"rollout_temperature": 0.7},
            },
            entrypoint="main.py",
            commit_sha="abcdef1234567890",
            env_config={"PROMPT_MODE": "strict"},
            resolved_secret_scopes={
                "OPENAI_API_KEY": "workspace",
                "ANTHROPIC_API_KEY": "user_override",
            },
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                raise PlatformAPIError("not available")

            get_training_run_metrics = _raise_metrics_unavailable

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(name="full-run")

        assert isinstance(result, DetailResult)
        fields = {field.label: field.value for field in result.fields}
        assert fields["Examples Processed"] == "100"
        assert fields["Notes"] == "experiment notes"
        assert fields["Submitted By"] == "alice"
        assert len(fields["Submitted"]) >= len("2025-12-31 00:00:00")
        assert len(fields["Started"]) >= len("2026-01-01 00:00:00")
        assert len(fields["Completed"]) >= len("2026-01-02 00:00:00")
        assert result.data["training_run"]["examples_processed_count"] == 100
        assert result.data["training_run"]["notes"] == "experiment notes"

        # Configuration lives in its own section, mirroring eval info.
        section_plain = {
            line.split(": ", 1)[0]: line.split(": ", 1)[1]
            for section in result.sections
            for line in section.plain_lines
            if ": " in line
        }
        assert section_plain["Entrypoint"] == "main.py"
        assert section_plain["Config"] == (
            "sampling.rollout_temperature=0.7, training.lr=0.001, "
            "training.total_epochs=3"
        )
        assert section_plain["Commit"] == "abcdef1"
        assert section_plain["Secrets"] == (
            "ANTHROPIC_API_KEY (personal, overrides workspace), "
            "OPENAI_API_KEY (workspace)"
        )
        assert section_plain["Environment Variables"] == "PROMPT_MODE=strict"


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


class TestLogs:
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
            step="train",
            message="OOM",
            details={"exit_code": 137},
        ),
    ]

    @staticmethod
    def _install_client(
        monkeypatch: pytest.MonkeyPatch, page: LogsPage
    ) -> dict[str, object]:
        captured: dict[str, object] = {}

        class FakeClient:
            def get_training_run_logs(
                self, name, *, limit, cursor=None, git_identity, credentials=None
            ):
                assert credentials is FAKE_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                captured["name"] = name
                captured["limit"] = limit
                captured["cursor"] = cursor
                return page

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        return captured

    def test_logs_renders_chronological_table(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = self._install_client(
            monkeypatch, LogsPage(logs=self.LOG_ENTRIES, next_cursor=None)
        )

        result = train_module.logs(name="run-1", limit=50, cursor=None)

        assert captured == {"name": "run-1", "limit": 50, "cursor": None}
        assert isinstance(result, ListResult)
        assert result.title == "Training Run Logs: run-1"
        assert [column.label for column in result.columns] == [
            "Time",
            "Level",
            "Step",
            "Message",
        ]
        # Oldest-first order is preserved from the server page.
        assert [item["message"] for item in result.items] == ["Run created", "OOM"]
        assert result.items[0]["timestamp"] == "2026-06-01T00:00:00Z"
        assert result.items[1]["details"] == {"exit_code": 137}
        assert result.total_count == 2
        assert result.has_more is False
        assert result.next_offset is None
        assert result.extra == {"next_cursor": None, **_git_extra()}
        assert result.display_items is not None
        # Display timestamps are localized, raw ISO stays in items for JSON.
        assert result.display_items[0]["timestamp"] != "2026-06-01T00:00:00Z"
        assert result.display_items[0]["timestamp"].startswith("2026-")
        assert result.display_hints == ["Use osmosis train info run-1 for run details."]

    def test_logs_has_more_when_next_cursor_present(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        self._install_client(
            monkeypatch,
            LogsPage(logs=self.LOG_ENTRIES, next_cursor="2026-06-01T00:00:00Z|log-1"),
        )

        result = train_module.logs(name="run-1", limit=2)

        assert isinstance(result, ListResult)
        assert result.has_more is True
        assert result.extra["next_cursor"] == "2026-06-01T00:00:00Z|log-1"

    def test_logs_passes_cursor_to_client(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = self._install_client(
            monkeypatch, LogsPage(logs=self.LOG_ENTRIES, next_cursor=None)
        )

        train_module.logs(name="run-1", limit=50, cursor="2026-06-01T00:00:00Z|log-1")

        assert captured["cursor"] == "2026-06-01T00:00:00Z|log-1"

    def test_logs_empty_page(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        self._install_client(monkeypatch, LogsPage(logs=[], next_cursor=None))

        result = train_module.logs(name="run-1", limit=50)

        assert isinstance(result, ListResult)
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False

    @pytest.mark.parametrize("limit", ["0", "201"])
    def test_logs_rejects_out_of_range_limit(self, limit: str, capsys) -> None:
        from osmosis_ai.cli.main import main

        rc = main(["train", "logs", "run-1", "--limit", limit])

        assert rc == 2
        assert "is not in the range 1<=x<=200" in capsys.readouterr().err

    def test_logs_unknown_run_emits_not_found_envelope(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class FakeClient:
            def get_training_run_logs(
                self, name, *, limit, cursor=None, git_identity, credentials=None
            ):
                raise PlatformAPIError("Training run not found", 404)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

        from osmosis_ai.cli import main as cli

        exit_code = cli.main(["--json", "train", "logs", "missing-run"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert captured.out == ""
        envelope = json.loads(captured.err)
        assert envelope["error"]["code"] == "NOT_FOUND"
        assert envelope["error"]["message"] == "Training run not found"
        assert envelope["command"] == "train logs"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    SUBMIT_RESULT = SubmitRunResult(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="my-training-run",
        status="pending",
        created_at="2026-04-10T12:00:00Z",
        platform_url="https://platform.osmosis.ai/ws/training/550e8400-e29b-41d4-a716-446655440000",
    )

    @staticmethod
    def _write_project(tmp_path: Path, *, rollout: str = "calculator") -> Path:
        return _make_workspace_directory(tmp_path, rollout=rollout)

    @classmethod
    def _write_config(cls, tmp_path: Path) -> Path:
        workspace_directory = cls._write_project(tmp_path)
        path = workspace_directory / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "calculator"
entrypoint = "main.py"
model_path = "Qwen/Qwen3.6-35B-A3B"
dataset = "abc-123"

[training]
lr = 1e-6
total_epochs = 1
n_samples_per_prompt = 8
rollout_batch_size = 64
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
        monkeypatch.chdir(config_path.parents[2])
        result = self.SUBMIT_RESULT

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return result

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        command_result = train_module.submit(config_path=config_path, yes=True)

        assert isinstance(command_result, OperationResult)
        assert command_result.operation == "train.submit"
        assert command_result.status == "success"
        assert command_result.resource is not None
        assert command_result.resource["name"] == "my-training-run"
        assert command_result.resource["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert command_result.resource["status"] == "pending"
        assert command_result.resource["model_name"] == "Qwen/Qwen3.6-35B-A3B"
        assert command_result.resource["dataset_name"] == "abc-123"
        assert "/training/" in command_result.resource["url"]
        assert command_result.resource["git"] == {
            "identity": GIT_IDENTITY,
            "remote_url": REPO_URL,
        }
        assert command_result.resource["workspace_directory"] == str(
            config_path.parents[2].resolve()
        )
        assert command_result.message == "Training run submitted: my-training-run"

    def test_submit_url_does_not_insert_rich_line_breaks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2])
        result = self.SUBMIT_RESULT

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return result

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        command_result = train_module.submit(config_path=config_path, yes=True)

        expected_url = result.platform_url
        assert isinstance(command_result, OperationResult)
        assert command_result.resource is not None
        assert command_result.resource["url"] == expected_url
        assert "Model: Qwen/Qwen3.6-35B-A3B" in command_result.display_next_steps
        assert "Dataset: abc-123" in command_result.display_next_steps
        assert f"View: {expected_url}" in command_result.display_next_steps

    def test_submit_shows_summary_table(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return self.SUBMIT_RESULT

        FakeClient.SUBMIT_RESULT = self.SUBMIT_RESULT
        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.submit(config_path=config_path, yes=True)
        out = console_capture.getvalue()
        assert "calculator" in out
        assert "main.py" in out
        assert "Qwen/Qwen3.6-35B-A3B" in out
        assert "abc-123" in out
        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["config"] == {
            "rollout": "calculator",
            "entrypoint": "main.py",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "dataset": "abc-123",
            "commit_sha": None,
        }

    def test_submit_with_commit_sha(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        workspace_directory = self._write_project(tmp_path, rollout="r")
        monkeypatch.chdir(workspace_directory)
        path = workspace_directory / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
commit_sha = "deadbeef"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
            encoding="utf-8",
        )

        captured_kwargs: dict = {}

        class FakeClient:
            def submit_training_run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.submit(config_path=path, yes=True)
        assert captured_kwargs["experiment_config"]["commit_sha"] == "deadbeef"
        assert captured_kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in captured_kwargs
        assert "deadbeef" in console_capture.getvalue()
        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["config"]["commit_sha"] == "deadbeef"

    def _write_commit_sha_config(self, tmp_path: Path) -> Path:
        workspace_directory = self._write_project(tmp_path, rollout="r")
        path = workspace_directory / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
commit_sha = "deadbeef"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
            encoding="utf-8",
        )
        return path

    def test_submit_aborts_on_invalid_pinned_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        path = self._write_commit_sha_config(tmp_path)
        monkeypatch.chdir(path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                raise AssertionError(
                    "submit must not run when the commit preflight fails"
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

        assert_submit_aborts_on_invalid_commit(
            monkeypatch,
            submit=lambda: train_module.submit(config_path=path, yes=True),
        )

    def test_submit_surfaces_pinned_commit_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        path = self._write_commit_sha_config(tmp_path)
        monkeypatch.chdir(path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

        assert_submit_surfaces_commit_warning(
            monkeypatch,
            submit=lambda: train_module.submit(config_path=path, yes=True),
            console_output=console_capture.getvalue,
        )

    def test_submit_passes_config_to_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2])
        captured_kwargs: dict = {}

        class FakeClient:
            def submit_training_run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)
        assert captured_kwargs["experiment_config"] == {
            "model_path": "Qwen/Qwen3.6-35B-A3B",
            "dataset": "abc-123",
            "rollout": "calculator",
            "entrypoint": "main.py",
        }
        assert captured_kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in captured_kwargs
        assert captured_kwargs["training_config"] == {
            "lr": 1e-6,
            "total_epochs": 1,
            "n_samples_per_prompt": 8,
            "rollout_batch_size": 64,
        }
        assert captured_kwargs["sampling_config"] is None
        assert captured_kwargs["checkpoints_config"] is None
        assert captured_kwargs["advanced_config"] is None

    def test_submit_passes_top_level_env_and_secrets_to_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        workspace_directory = self._write_project(tmp_path, rollout="r")
        monkeypatch.chdir(workspace_directory)
        path = workspace_directory / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"

[env]
LOG_LEVEL = "INFO"

[secrets]
required = ["OPENAI_API_KEY"]
""".strip(),
            encoding="utf-8",
        )
        captured_kwargs: dict = {}

        class FakeClient:
            def submit_training_run(self, **kwargs):
                captured_kwargs.update(kwargs)
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=path, yes=True)

        assert captured_kwargs["env_config"] == {"LOG_LEVEL": "INFO"}
        assert captured_kwargs["secrets"] == ["OPENAI_API_KEY"]

    def test_submit_rejects_non_canonical_training_config_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._write_project(tmp_path)
        monkeypatch.chdir(tmp_path)
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
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        workspace_directory = self._write_project(tmp_path, rollout="graderless")
        monkeypatch.chdir(workspace_directory)
        (workspace_directory / "rollouts" / "graderless" / "main.py").write_text(
            """
from osmosis_ai.rollout import AgentWorkflow


class TestWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None
""".strip(),
            encoding="utf-8",
        )
        path = workspace_directory / "configs" / "training" / "graderless.toml"
        path.write_text(
            """
[experiment]
rollout = "graderless"
entrypoint = "main.py"
model_path = "m"
dataset = "d"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
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
        monkeypatch.chdir(config_path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=config_path, yes=True)

        out = console_capture.getvalue()
        assert "Platform source selection may differ from your local branch" in out
        assert "Platform-connected repository" in out
        assert "committed and pushed" in out

    def test_submit_warns_about_unpushed_commits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = self._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2])

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
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
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
        monkeypatch.chdir(config_path.parents[2])

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
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
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
        monkeypatch.chdir(config_path.parents[2])

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
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
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
        workspace_directory = self._write_project(tmp_path, rollout="r")
        monkeypatch.chdir(workspace_directory)
        path = workspace_directory / "configs" / "training" / "train.toml"
        path.write_text(
            """
[experiment]
rollout = "r"
entrypoint = "main.py"
model_path = "m"
dataset = "d"
commit_sha = "deadbeef1234"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
            encoding="utf-8",
        )

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        train_module.submit(config_path=path, yes=True)

        out = console_capture.getvalue()
        assert "deadbeef1234" in out
        assert "Platform-connected repository" in out
        assert "pushed to origin" in out

    def test_submit_accepts_project_subdirectory_cwd(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config_path = self._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2] / "rollouts" / "calculator")

        class FakeClient:
            def submit_training_run(self, **kwargs):
                assert kwargs["git_identity"] == GIT_IDENTITY
                assert "workspace_id" not in kwargs
                return TestSubmit.SUBMIT_RESULT

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.submit(config_path=config_path, yes=True)

        assert isinstance(result, OperationResult)
        assert result.resource is not None
        assert result.resource["workspace_directory"] == str(
            config_path.parents[2].resolve()
        )


def test_train_submit_resolves_project_from_cwd_not_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from osmosis_ai.cli.main import main

    outside = tmp_path / "outside"
    outside.mkdir()
    config = outside / "default.toml"
    config.write_text(
        "[experiment]\nrollout='demo'\nentrypoint='main.py'\nmodel_path='m'\ndataset='ds'\n",
        encoding="utf-8",
    )
    project = _make_workspace_directory(tmp_path / "project")
    monkeypatch.chdir(project)

    rc = main(["--json", "train", "submit", str(config), "--yes"])

    assert rc == 1
    assert "configs/training" in capsys.readouterr().err


def test_train_submit_requires_linked_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from osmosis_ai.cli.main import main

    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "training" / "default.toml"
    config.write_text(
        (
            "[experiment]\n"
            "rollout='demo'\n"
            "entrypoint='main.py'\n"
            "model_path='m'\n"
            "dataset='ds'\n\n"
            "[training]\n"
            "n_samples_per_prompt=1\n"
            "rollout_batch_size=1\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    rc = main(["--json", "train", "submit", "configs/training/default.toml", "--yes"])

    assert rc == 1
    assert (
        "Set `origin` to the Platform-connected repository" in capsys.readouterr().err
    )


def test_train_submit_accepts_project_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from osmosis_ai.cli.main import main

    project = _make_workspace_directory(tmp_path / "project")
    config = project / "configs" / "training" / "default.toml"
    config.write_text(
        (
            "[experiment]\n"
            "rollout='demo'\n"
            "entrypoint='main.py'\n"
            "model_path='m'\n"
            "dataset='ds'\n\n"
            "[training]\n"
            "n_samples_per_prompt=1\n"
            "rollout_batch_size=1\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project / "rollouts" / "demo")

    class FakeClient:
        def submit_training_run(self, **kwargs):
            assert kwargs["git_identity"] == GIT_IDENTITY
            assert "workspace_id" not in kwargs
            return TestSubmit.SUBMIT_RESULT

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
    rc = main(["--json", "train", "submit", "configs/training/default.toml", "--yes"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["resource"]["workspace_directory"] == str(project.resolve())


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
        monkeypatch.chdir(config_path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                raise AssertionError("API must not be reached without confirmation")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

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
            "Model": "Qwen/Qwen3.6-35B-A3B",
            "Dataset": "abc-123",
        }
        assert any("Platform-connected repository" in note for note in details["notes"])
        assert any(
            "Platform source selection may differ from your local branch" in warning
            for warning in details["warnings"]
        )

    def test_submit_json_includes_git_warnings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from osmosis_ai.platform.cli import workspace_repo

        config_path = TestSubmit._write_config(tmp_path)
        monkeypatch.chdir(config_path.parents[2])

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
        monkeypatch.chdir(config_path.parents[2])

        class FakeClient:
            def submit_training_run(self, **kwargs):
                raise AssertionError("API must not be reached without confirmation")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)

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
        from osmosis_ai.cli.prompts import require_confirmation

        with override_output_context(format=OutputFormat.plain, interactive=False):
            with pytest.raises(CLIError) as exc_info:
                require_confirmation(
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
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_passes_git_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        console_capture: StringIO,
        tmp_path: Path,
    ) -> None:
        detail = TrainingRunDetail(
            id="abcdef1234567890abcdef1234567890",
            name="run-1",
            status="completed",
            model_name="gpt-2",
            platform_url="https://platform.osmosis.ai/ws/training/abcdef1234567890abcdef1234567890",
        )
        metrics = TrainingRunMetrics(
            training_run_id=detail.id,
            status="completed",
            overview=TrainingRunMetricsOverview(
                duration_ms=None,
                metric_summaries=[],
                examples_processed_count=None,
            ),
            metrics=[],
        )

        class FakeClient:
            def get_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return detail

            def get_training_run_metrics(
                self, run_id, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                return metrics

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.info(
            name="run-1",
            output=str(tmp_path / "metrics.json"),
        )

        assert isinstance(result, DetailResult)
        assert result.title == "Training Run Info"
        assert result.data["platform_url"] == detail.platform_url
        assert {
            key: result.data[key] for key in ("git", "workspace_directory")
        } == _git_extra()


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_basic(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def stop_training_run(self, run_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
                return {"stopped": True}

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(platform_train_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(shared_submit_module, "OsmosisClient", FakeClient)
        result = train_module.stop(name="my-run", yes=True)

        assert isinstance(result, OperationResult)
        assert result.operation == "train.stop"
        assert result.status == "success"
        assert result.resource == {"name": "my-run", **_git_extra()}
        assert result.message == 'Training run "my-run" stopped.'
