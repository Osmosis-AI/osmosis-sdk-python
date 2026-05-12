"""CommandResult smoke tests for train/model/deployment/rollout commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from osmosis_ai.cli import main as cli
from osmosis_ai.platform.api.models import (
    BaseModelInfo,
    DeploymentInfo,
    DeploymentSummary,
    MetricDataPoint,
    MetricHistory,
    PaginatedBaseModels,
    PaginatedDeployments,
    PaginatedRollouts,
    PaginatedTrainingRuns,
    RolloutInfo,
    SubmitTrainingRunResult,
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
    project_root: Path = GIT_PROJECT_ROOT,
) -> None:
    assert payload["project_root"] == str(project_root.resolve())
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
        project_root = (
            Path.cwd().resolve()
            if (Path.cwd() / ".osmosis").is_dir()
            else GIT_PROJECT_ROOT
        )
        return SimpleNamespace(
            project_root=project_root,
            git_identity=GIT_IDENTITY,
            repo_url=REPO_URL,
            credentials=FAKE_CREDENTIALS,
        )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_project_context",
        _git_context,
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
        ".osmosis/research",
        "rollouts/demo",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "research" / "program.md").write_text(
        "# Test Program\n", encoding="utf-8"
    )
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
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "train", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["items"][0]["name"] == "reward-run"
    assert payload["total_count"] == 1
    _assert_git_context(payload)
    assert captured.out.count("\n") == 1


def test_train_status_json_returns_detail_envelope(
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
                status="running",
                model_name="Qwen/Qwen3",
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "train", "status", "reward-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["name"] == "reward-run"
    assert payload["data"]["status"] == "running"
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
            assert kwargs["rollout_name"] == "demo"
            return SubmitTrainingRunResult(
                id="run_1",
                name="reward-run",
                status="pending",
                created_at="2026-04-26T00:00:00Z",
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "train", "submit", str(config_path), "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "train.submit"
    assert payload["resource"]["name"] == "reward-run"
    _assert_git_context(payload["resource"], project)


def test_train_metrics_json_does_not_write_default_file(
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
        ".osmosis/research",
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
                    mlflow_run_id="mlflow_1",
                    mlflow_status="FINISHED",
                    duration_ms=1000,
                    duration_formatted="1s",
                    reward=1.0,
                    reward_delta=0.5,
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

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "train", "metrics", "reward-run"])
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

    exit_code = cli.main(["--json", "train", "stop", "reward-run", "--yes"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "train.stop"
    assert payload["resource"]["name"] == "reward-run"
    _assert_git_context(payload["resource"])


def test_model_list_plain_is_tab_separated_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

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
                        status="ready",
                        creator_name="brian",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--plain", "model", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == [
        "Qwen/Qwen3\tQwen/Qwen3\tready\tbrian\t2026-04-26T00:00:00Z\tmodel_1"
    ]


def test_model_list_json_includes_git_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

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
                        status="ready",
                        creator_name="brian",
                        created_at="2026-04-26T00:00:00Z",
                    )
                ],
                total_count=1,
                has_more=False,
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "model", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["items"][0]["model_name"] == "Qwen/Qwen3"
    _assert_git_context(payload)


def test_deployment_commands_json_return_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def list_deployments(
            self, limit=30, offset=0, *, git_identity, credentials=None
        ):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return PaginatedDeployments(
                deployments=[
                    DeploymentInfo(
                        id="dep_1",
                        checkpoint_name="run-step-1",
                        status="active",
                        checkpoint_step=1,
                        base_model="Qwen/Qwen3",
                    )
                ],
                total_count=1,
                has_more=False,
            )

        def get_deployment(self, checkpoint, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return DeploymentInfo(
                id="dep_1",
                checkpoint_name=checkpoint,
                status="active",
                checkpoint_step=1,
                base_model="Qwen/Qwen3",
                training_run_name="reward-run",
            )

        def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return DeploymentSummary(
                id="dep_1", checkpoint_name=checkpoint, status="active"
            )

        def undeploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return DeploymentSummary(
                id="dep_1", checkpoint_name=checkpoint, status="inactive"
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "deployment", "list"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["items"][0]["checkpoint_name"] == "run-step-1"
    _assert_git_context(payload)

    exit_code = cli.main(["--json", "deployment", "info", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["checkpoint_name"] == "run-step-1"
    _assert_git_context(payload["data"])

    exit_code = cli.main(["--json", "deploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "deploy"
    assert payload["resource"]["checkpoint_name"] == "run-step-1"
    _assert_git_context(payload["resource"])

    exit_code = cli.main(["--json", "undeploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "undeploy"
    assert payload["resource"]["checkpoint_name"] == "run-step-1"
    assert payload["resource"]["status"] == "inactive"
    _assert_git_context(payload["resource"])


def test_failed_deploy_json_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_git_context(monkeypatch)

    class FakeClient:
        def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
            assert credentials is FAKE_CREDENTIALS
            assert git_identity == GIT_IDENTITY
            return DeploymentSummary(
                id="dep_1",
                checkpoint_name=checkpoint,
                status="failed",
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "deploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["operation"] == "deploy"
    assert payload["status"] == "failed"
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

    exit_code = cli.main(["--json", "rollout", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["items"][0]["name"] == "demo"
    assert payload["next_offset"] is None
    _assert_git_context(payload)
