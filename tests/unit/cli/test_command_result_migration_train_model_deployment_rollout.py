"""CommandResult smoke tests for train/model/deployment/rollout commands."""

from __future__ import annotations

import json
from pathlib import Path

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
    TrainingRun,
    TrainingRunDetail,
    TrainingRunMetrics,
    TrainingRunMetricsOverview,
)


def _stub_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


def _make_rollout_project(root: Path) -> Path:
    for rel_path in (
        ".osmosis/research",
        "rollouts/demo",
        "configs/training",
        "configs/eval",
        "data",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / ".osmosis" / "project.toml").write_text(
        "[project]\nsetup_source = 'test'\n", encoding="utf-8"
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
    _stub_auth(monkeypatch)

    class FakeClient:
        def list_training_runs(self, limit=30, offset=0, *, credentials=None):
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
    assert captured.out.count("\n") == 1


def test_train_metrics_json_does_not_write_default_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _stub_auth(monkeypatch)
    osmosis_dir = tmp_path / ".osmosis"
    osmosis_dir.mkdir()
    (osmosis_dir / "project.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        def get_training_run(self, name, *, credentials=None):
            return TrainingRunDetail(
                id="run_1",
                name=name,
                status="finished",
                model_name="Qwen/Qwen3",
            )

        def get_training_run_metrics(self, run_id, *, credentials=None):
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
    assert not (osmosis_dir / "metrics").exists()


def test_train_traces_json_returns_structured_not_implemented_error(capsys) -> None:
    exit_code = cli.main(["--json", "train", "traces"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "NOT_IMPLEMENTED"
    assert "train traces" in payload["error"]["message"]


def test_model_list_plain_is_tab_separated_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_auth(monkeypatch)

    class FakeClient:
        def list_base_models(self, limit=30, offset=0, *, credentials=None):
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


def test_deployment_commands_json_return_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_auth(monkeypatch)

    class FakeClient:
        def list_deployments(self, limit=30, offset=0, *, credentials=None):
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

        def deploy_checkpoint(self, checkpoint, *, credentials=None):
            return DeploymentSummary(
                id="dep_1", checkpoint_name=checkpoint, status="active"
            )

    monkeypatch.setattr("osmosis_ai.platform.api.client.OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "deployment", "list"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["items"][0]["checkpoint_name"] == "run-step-1"

    exit_code = cli.main(["--json", "deploy", "run-step-1"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "deploy"
    assert payload["resource"]["checkpoint_name"] == "run-step-1"


def test_failed_deploy_json_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_auth(monkeypatch)

    class FakeClient:
        def deploy_checkpoint(self, checkpoint, *, credentials=None):
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


def test_destructive_command_requires_yes_in_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_auth(monkeypatch)

    exit_code = cli.main(["--json", "deployment", "delete", "run-step-1"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "INTERACTIVE_REQUIRED"


def test_rollout_list_json_returns_envelope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_auth(monkeypatch)

    class FakeClient:
        def list_rollouts(self, limit=30, offset=0, *, credentials=None):
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


def test_rollout_validate_json_returns_detail_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(["--json", "rollout", "validate", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is True
    assert payload["data"]["kind"] == "training"
