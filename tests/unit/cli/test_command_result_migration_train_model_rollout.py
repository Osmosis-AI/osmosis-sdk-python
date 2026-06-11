"""CommandResult smoke tests for train/model/rollout commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    LoraModelInfo,
    LoraModelSummary,
    MetricDataPoint,
    MetricHistory,
    PaginatedBaseModels,
    PaginatedLoraModels,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    RolloutInfo,
    SubmitRunResult,
    TrainingRun,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
GIT_PROJECT_ROOT = Path("/repo")
FAKE_CREDENTIALS = object()


def _assert_git_context(
    payload: dict[str, object],
    workspace_directory: Path = GIT_PROJECT_ROOT,
) -> None:
    assert payload["workspace_directory"] == str(workspace_directory.resolve())
    assert payload["git"] == {
        "identity": GIT_IDENTITY,
        "remote_url": REPO_URL,
    }
    assert "workspace" not in payload


def _stub_git_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def _git_top_level(start: Path) -> Path | None:
        current = start.resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".osmosis").is_dir():
                return candidate
        return None

    def _git_context() -> SimpleNamespace:
        workspace_directory = (
            Path.cwd().resolve()
            if (Path.cwd() / ".osmosis").is_dir()
            else GIT_PROJECT_ROOT
        )
        return SimpleNamespace(
            workspace_directory=workspace_directory,
            git_identity=GIT_IDENTITY,
            repo_url=REPO_URL,
            credentials=FAKE_CREDENTIALS,
        )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_workspace_directory_context",
        _git_context,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.require_git_workspace_directory_context",
        _git_context,
    )
    for _delegated in ("model", "rollout"):
        monkeypatch.setattr(
            f"osmosis_ai.platform.cli.{_delegated}."
            "require_git_workspace_directory_context",
            _git_context,
            raising=False,
        )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.shared_submit.require_git_workspace_directory_context",
        _git_context,
        raising=False,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.git_worktree_top_level",
        _git_top_level,
    )


def _make_rollout_project(root: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis",
        "rollouts/demo",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / "demo" / "main.py").write_text(
        """
from osmosis_ai.rollout.agent_workflow import AgentWorkflow
from osmosis_ai.rollout.grader import Grader


class DemoWorkflow(AgentWorkflow):
    async def run(self, ctx):
        return None


class DemoGrader(Grader):
    async def grade(self, ctx):
        return 1.0
""".strip(),
        encoding="utf-8",
    )
    return root


def test_train_list_json_returns_single_list_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_training_runs(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return PaginatedTrainingRuns(
                training_runs=[
                    TrainingRun(
                        id="run_1",
                        name="reward-run",
                        status="running",
                        model_name="Qwen/Qwen3",
                        rollout_id="rollout_1",
                        rollout_name="math-rollout",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "train", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["items"][0]["name"] == "reward-run"
    assert payload["items"][0]["rollout_name"] == "math-rollout"
    assert payload["total_count"] == 1
    _assert_git_context(payload)
    assert captured.out.count("\n") == 1


def test_train_info_json_returns_combined_detail_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def get_training_run(self, name, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return TrainingRunDetail(
                id="run_1",
                name=name,
                status="pending",
                model_name="Qwen/Qwen3",
                rollout_id="rollout_1",
                rollout_name="math-rollout",
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "train", "info", "reward-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["training_run"]["name"] == "reward-run"
    assert payload["data"]["training_run"]["status"] == "pending"
    assert payload["data"]["training_run"]["rollout_name"] == "math-rollout"
    assert payload["data"]["checkpoints"] == []
    assert payload["data"]["metrics_available"] is False
    assert payload["data"]["output_path"] is None
    _assert_git_context(payload["data"])


def test_train_submit_json_returns_operation_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _make_rollout_project(tmp_path)
    config_path = project / "configs" / "training" / "demo.toml"
    config_path.write_text(
        """
[experiment]
rollout = "demo"
entrypoint = "main.py"
model_path = "Qwen/Qwen3"
dataset = "demo-dataset"

[training]
n_samples_per_prompt = 8
rollout_batch_size = 64
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    _stub_git_context(monkeypatch)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.require_git_top_level",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.workspace_repo.summarize_local_git_state",
        lambda _root: None,
    )

    class FakeClient:
        def submit_training_run(self, **kwargs):
            assert kwargs["credentials"] is FAKE_CREDENTIALS
            assert kwargs["git_identity"] == GIT_IDENTITY
            assert "workspace_id" not in kwargs
            assert kwargs["experiment_config"]["rollout"] == "demo"
            return SubmitRunResult(
                id="run_1",
                name="reward-run",
                status="pending",
                created_at="2026-04-26T00:00:00Z",
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.OsmosisClient", FakeClient, raising=False
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.shared_submit.OsmosisClient",
        FakeClient,
        raising=False,
    )

    exit_code = cli.main(["--json", "train", "submit", str(config_path), "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "train.submit"
    assert payload["resource"]["name"] == "reward-run"
    assert payload["resource"]["model_name"] == "Qwen/Qwen3"
    assert payload["resource"]["dataset_name"] == "demo-dataset"
    _assert_git_context(payload["resource"], project)


def test_train_info_json_does_not_write_default_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _stub_git_context(monkeypatch)
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for rel_path in (
        ".osmosis",
        "rollouts",
        "configs/eval",
        "configs/training",
        "data",
    ):
        (tmp_path / rel_path).mkdir(parents=True, exist_ok=True)
    osmosis_dir = tmp_path / ".osmosis"
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        def get_training_run(self, name, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return TrainingRunDetail(
                id="run_1",
                name=name,
                status="finished",
                model_name="Qwen/Qwen3",
            )

        def get_training_run_metrics(self, run_id, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return TrainingRunMetrics(
                training_run_id=run_id,
                status="finished",
                overview=TrainingRunMetricsOverview(
                    duration_ms=1000,
                    metric_summaries=[],
                    examples_processed_count=10,
                ),
                metrics=[
                    MetricHistory(
                        metric_key="rollout/raw_reward",
                        title="Reward",
                        data_points=[MetricDataPoint(step=1, value=1.0, timestamp=0)],
                    )
                ],
            )

        def list_training_run_checkpoints(
            self, run_id, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return type("CheckpointPage", (), {"checkpoints": []})()

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "train", "info", "reward-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["metrics_available"] is True
    assert payload["data"]["output_path"] is None
    _assert_git_context(payload["data"], tmp_path)
    assert not (osmosis_dir / "metrics").exists()


def test_train_stop_json_returns_operation_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def stop_training_run(self, name, *, git_identity, credentials=None):
            assert name == "reward-run"
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return {"stopped": True}

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.train.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "train", "stop", "reward-run", "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "train.stop"
    assert payload["resource"]["name"] == "reward-run"
    _assert_git_context(payload["resource"])


def _model_list_fake_client():
    class FakeClient:
        def list_base_models(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return PaginatedBaseModels(
                models=[
                    BaseModelInfo(
                        id="model_1",
                        model_name="Qwen/Qwen3",
                        base_model="Qwen/Qwen3",
                        creator_name="brian",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

        def list_lora_models(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return PaginatedLoraModels(
                models=[
                    LoraModelInfo(
                        id="lora_1",
                        model_name="run-step-1",
                        base_model="Qwen/Qwen3",
                        training_run_name="reward-run",
                        checkpoint_step=1,
                        reward=0.87,
                        deployment_status="active",
                        created_at="2026-04-27T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    return FakeClient


def test_model_list_plain_outputs_titled_base_and_lora_sections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)
    FakeClient = _model_list_fake_client()

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.model.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--plain", "model", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    lines = captured.out.splitlines()
    assert len(lines) == 4
    assert lines[0] == "Base Models:"
    assert len(lines[1].split("\t")) == 3
    assert lines[1].startswith("Qwen/Qwen3\t")
    assert lines[1].endswith("\tbrian")
    assert lines[2] == "LoRA Models:"
    assert len(lines[3].split("\t")) == 7
    assert lines[3].startswith("run-step-1\tQwen/Qwen3\treward-run\t1\t0.87\t")
    assert lines[3].endswith("\t[deployed]")


def test_model_list_json_returns_sectioned_list_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)
    FakeClient = _model_list_fake_client()

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.model.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "model", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert "items" not in payload
    assert "total_count" not in payload

    base_models = payload["base_models"]
    assert base_models["total_count"] == 1
    assert base_models["has_more"] is False
    assert base_models["next_offset"] is None
    assert [i["model_name"] for i in base_models["items"]] == ["Qwen/Qwen3"]
    assert "type" not in base_models["items"][0]
    assert "status" not in base_models["items"][0]

    lora_models = payload["lora_models"]
    assert lora_models["total_count"] == 1
    assert lora_models["has_more"] is False
    assert lora_models["next_offset"] is None
    assert [i["model_name"] for i in lora_models["items"]] == ["run-step-1"]
    assert "type" not in lora_models["items"][0]
    assert lora_models["items"][0]["deployment_status"] == "active"
    assert lora_models["items"][0]["checkpoint_step"] == 1
    assert lora_models["items"][0]["reward"] == 0.87
    _assert_git_context(payload)
    assert captured.out.count("\n") == 1


def test_model_deploy_undeploy_json_return_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def deploy_lora_model(self, lora_model_name, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return LoraModelSummary(
                id="lora_1", model_name=lora_model_name, status="active"
            )

        def undeploy_lora_model(
            self, lora_model_name, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return LoraModelSummary(
                id="lora_1", model_name=lora_model_name, status="inactive"
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.model.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "model", "deploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "model.deploy"
    assert payload["resource"]["model_name"] == "run-step-1"
    _assert_git_context(payload["resource"])

    exit_code = cli.main(["--json", "model", "undeploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "model.undeploy"
    assert payload["resource"]["model_name"] == "run-step-1"
    assert payload["resource"]["status"] == "inactive"
    _assert_git_context(payload["resource"])


def test_rollout_list_json_returns_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_rollouts(self, limit=30, offset=0, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return PaginatedRollouts(
                rollouts=[
                    RolloutInfo(
                        id="rollout_1",
                        name="demo",
                        is_active=True,
                        repo_full_name="osmosis/demo",
                        last_synced_commit_sha="abc123",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.rollout.OsmosisClient", FakeClient, raising=False
    )

    exit_code = cli.main(["--json", "rollout", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["items"][0]["name"] == "demo"
    assert payload["next_offset"] is None
    _assert_git_context(payload)


def test_rollout_list_columns_prioritize_name_over_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-04-26T00:00:00Z"
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_rollouts(self, limit=30, offset=0, *, git_identity, credentials=None):
            assert git_identity == GIT_IDENTITY
            return PaginatedRollouts(
                rollouts=[
                    RolloutInfo(
                        id="rollout_1",
                        name="demo",
                        is_active=True,
                        repo_full_name="osmosis/demo",
                        last_synced_commit_sha="abcdef123456",
                        created_at=created_at,
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.rollout.OsmosisClient", FakeClient, raising=False
    )

    from osmosis_ai.cli.commands.rollout import list_rollouts
    from osmosis_ai.cli.output.display import format_local_date

    result = list_rollouts(limit=30, all_=False)

    assert [column.key for column in result.columns] == [
        "name",
        "is_active",
        "last_synced_commit_sha",
        "created_at",
    ]
    name_column = result.columns[0]
    assert name_column.ratio == 6
    assert name_column.overflow == "fold"
    assert name_column.min_width == 20
    assert result.columns[1].no_wrap is True
    assert result.columns[1].min_width == 6
    assert result.columns[1].max_width == 6
    assert result.columns[2].max_width == 8
    assert result.columns[3].max_width == 10
    assert result.items[0]["last_synced_commit_sha"] == "abcdef123456"
    assert result.display_items is not None
    assert result.display_items[0]["is_active"] == "yes"
    assert result.display_items[0]["last_synced_commit_sha"] == "abcdef12"
    assert (
        result.display_items[0]["created_at"]
        == format_local_date(created_at).split(" ", 1)[0]
    )
