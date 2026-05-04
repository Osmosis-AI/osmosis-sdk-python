"""Tests for osmosis_ai.cli.commands.deployment."""

from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.commands.deployment as deployment_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    DetailResult,
    ListResult,
    OperationResult,
    OutputFormat,
    override_output_context,
)
from osmosis_ai.platform.api.models import (
    DeploymentInfo,
    DeploymentSummary,
    LoraCheckpointInfo,
    PaginatedDeployments,
    PaginatedTrainingRuns,
    RenameDeploymentResult,
    TrainingRun,
    TrainingRunCheckpoints,
)

AUTH_CREDENTIALS = object()
WORKSPACE_ID = "ws-test"
WORKSPACE_NAME = "team-test"
PROJECT_ROOT = Path("/tmp/osmosis-project")


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(deployment_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture()
def mock_workspace_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    workspace = SimpleNamespace(
        project_root=PROJECT_ROOT,
        workspace_id=WORKSPACE_ID,
        workspace_name=WORKSPACE_NAME,
        credentials=AUTH_CREDENTIALS,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_workspace_context",
        lambda: workspace,
    )
    return workspace


@pytest.fixture()
def linked_project(mock_workspace_context: None) -> None:
    """Alias the linked project context expected by CLI-level deployment tests."""


def _deployment_summary(checkpoint: str) -> DeploymentSummary:
    return DeploymentSummary(
        id="dep_1",
        checkpoint_name=checkpoint,
        status="active",
    )


def test_deploy_without_checkpoint_requires_rich_interactive(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    rc = main(["--json", "deploy"])

    assert rc == 1
    assert "INTERACTIVE_REQUIRED" in capsys.readouterr().err


def test_deploy_with_checkpoint_still_works(
    monkeypatch: pytest.MonkeyPatch, linked_project, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.setattr(
        "osmosis_ai.platform.api.client.OsmosisClient.deploy_checkpoint",
        lambda self, checkpoint, *, credentials, workspace_id: _deployment_summary(
            checkpoint
        ),
    )

    rc = main(["--json", "deploy", "ckpt_1"])

    assert rc == 0


def test_deployment_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "deployment", "list"])

    assert rc == 1
    assert "Not in an Osmosis project" in capsys.readouterr().err


class TestDeployWizardHelper:
    def test_select_checkpoint_returns_selected_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")
        checkpoint = LoraCheckpointInfo(
            id="cp_1",
            checkpoint_name="reward-run-step-100",
            checkpoint_step=100,
            status="ready",
        )
        calls: list[tuple[str, object]] = []

        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                calls.append(("runs", workspace_id))
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, workspace_id, credentials=None
            ):
                calls.append(("checkpoints", workspace_id))
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name="reward-run",
                    checkpoints=[checkpoint],
                )

        def fake_select_list(message, *, items, actions):
            assert items
            if message == "Choose a training run":
                return run
            if message == "Choose a checkpoint":
                return checkpoint
            raise AssertionError(f"unexpected prompt: {message}")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        selected = deployment_module._select_checkpoint_for_deploy(
            mock_workspace_context
        )

        assert selected == "reward-run-step-100"
        assert calls == [("runs", WORKSPACE_ID), ("checkpoints", WORKSPACE_ID)]

    def test_select_checkpoint_cancel_at_run(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")

        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, workspace_id, credentials=None
            ):
                raise AssertionError("checkpoint list should not be fetched")

        def fake_select_list(message, *, items, actions):
            assert message == "Choose a training run"
            assert items
            return None

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        with pytest.raises(CLIError, match="Deploy cancelled"):
            deployment_module._select_checkpoint_for_deploy(mock_workspace_context)

    def test_select_checkpoint_cancel_at_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")
        checkpoint = LoraCheckpointInfo(
            id="cp_1",
            checkpoint_name="reward-run-step-100",
            checkpoint_step=100,
            status="ready",
        )

        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, workspace_id, credentials=None
            ):
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name="reward-run",
                    checkpoints=[checkpoint],
                )

        def fake_select_list(message, *, items, actions):
            assert items
            if message == "Choose a training run":
                return run
            if message == "Choose a checkpoint":
                return None
            raise AssertionError(f"unexpected prompt: {message}")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        with pytest.raises(CLIError, match="Deploy cancelled"):
            deployment_module._select_checkpoint_for_deploy(mock_workspace_context)

    def test_select_checkpoint_back_returns_to_run_selection(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        first_run = TrainingRun(id="run_1", name="first-run", status="finished")
        second_run = TrainingRun(id="run_2", name="second-run", status="finished")
        checkpoint = LoraCheckpointInfo(
            id="cp_2",
            checkpoint_name="second-run-step-100",
            checkpoint_step=100,
            status="ready",
        )
        selected_runs = iter([first_run, second_run])
        selected_checkpoints = iter(["__back__", checkpoint])
        checkpoint_calls: list[str] = []

        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[first_run, second_run],
                    total_count=2,
                    has_more=False,
                )

            def list_training_run_checkpoints(
                self, run_id, *, workspace_id, credentials=None
            ):
                checkpoint_calls.append(run_id)
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name=run_id,
                    checkpoints=[checkpoint],
                )

        def fake_select_list(message, *, items, actions):
            assert items
            if message == "Choose a training run":
                return next(selected_runs)
            if message == "Choose a checkpoint":
                return next(selected_checkpoints)
            raise AssertionError(f"unexpected prompt: {message}")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        selected = deployment_module._select_checkpoint_for_deploy(
            mock_workspace_context
        )

        assert selected == "second-run-step-100"
        assert checkpoint_calls == ["run_1", "run_2"]

    def test_select_checkpoint_no_training_runs_raises_before_prompt(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

        def fake_select_list(message, *, items, actions):
            raise AssertionError("select_list should not be called")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        with pytest.raises(CLIError, match="No training runs"):
            deployment_module._select_checkpoint_for_deploy(mock_workspace_context)

    def test_select_checkpoint_no_checkpoints_raises_before_checkpoint_prompt(
        self, monkeypatch: pytest.MonkeyPatch, mock_workspace_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")

        class FakeClient:
            def list_training_runs(self, *, workspace_id, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, workspace_id, credentials=None
            ):
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name="reward-run",
                    checkpoints=[],
                )

        def fake_select_list(message, *, items, actions):
            if message == "Choose a training run":
                assert items
                return run
            raise AssertionError("checkpoint prompt should not be shown")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        with pytest.raises(CLIError, match="No deployable checkpoints"):
            deployment_module._select_checkpoint_for_deploy(mock_workspace_context)


@pytest.mark.usefixtures("mock_workspace_context")
class TestListDeployments:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_deployments(
                self, limit=30, offset=0, *, workspace_id, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert workspace_id == WORKSPACE_ID
                return PaginatedDeployments(
                    deployments=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.list_deployments(limit=30, all_=False)

        assert isinstance(result, ListResult)
        assert result.title == "Deployments"
        assert result.items == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.extra == {
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }

    def test_list_with_deployments(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}
        dep = DeploymentInfo(
            id="dep_1",
            checkpoint_name="qwen3-run1-step-100",
            status="active",
            checkpoint_step=100,
            base_model="Qwen/Qwen3",
            created_at="2026-04-20T00:00:00Z",
        )

        class FakeClient:
            def list_deployments(
                self, limit=30, offset=0, *, workspace_id, credentials=None
            ):
                captured["limit"] = limit
                captured["offset"] = offset
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return PaginatedDeployments(
                    deployments=[dep], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.list_deployments(limit=10, all_=False)
        assert captured == {
            "limit": 10,
            "offset": 0,
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, ListResult)
        assert result.title == "Deployments"
        assert result.items[0]["checkpoint_name"] == "qwen3-run1-step-100"
        assert result.items[0]["base_model"] == "Qwen/Qwen3"


@pytest.mark.usefixtures("mock_workspace_context")
class TestInfo:
    def test_info_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def get_deployment(self, checkpoint, *, workspace_id, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return DeploymentInfo(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="active",
                    checkpoint_step=100,
                    base_model="Qwen/Qwen3",
                    training_run_name="qwen3-run1",
                    creator_name="brian",
                    created_at="2026-04-20T00:00:00Z",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.info(checkpoint=checkpoint_id)
        assert captured == {
            "checkpoint": checkpoint_id,
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, DetailResult)
        assert result.title == "Deployment"
        assert result.data["checkpoint_name"] == "qwen3-run1-step-100"
        assert result.data["base_model"] == "Qwen/Qwen3"
        assert result.data["training_run_name"] == "qwen3-run1"
        assert result.data["workspace"] == {
            "id": WORKSPACE_ID,
            "name": WORKSPACE_NAME,
        }
        assert result.data["project_root"] == str(PROJECT_ROOT)

    def test_info_escapes_status_for_rich_table(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_deployment(self, checkpoint, *, workspace_id, credentials=None):
                assert workspace_id == WORKSPACE_ID
                return DeploymentInfo(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="[red]failed[/red]",
                    checkpoint_step=100,
                    base_model="Qwen/Qwen3",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        result = deployment_module.info(checkpoint="qwen3-run1-step-100")

        assert isinstance(result, DetailResult)
        assert ("Status", "\\[red]failed\\[/red]") in [
            (field.label, field.value) for field in result.fields
        ]


@pytest.mark.usefixtures("mock_workspace_context")
class TestDeploy:
    def test_deploy_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, workspace_id, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="active",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.deploy(checkpoint="qwen3-run1-step-100")
        assert captured == {
            "checkpoint": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "deploy"
        assert result.status == "success"
        assert result.resource == {
            "id": "dep_1",
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "active",
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }
        assert result.message == "Deployment qwen3-run1-step-100 active"

    def test_deploy_escapes_checkpoint_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, workspace_id, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["workspace_id"] = workspace_id
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="active",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        with override_output_context(
            format=OutputFormat.rich, interactive=True
        ) as output:
            monkeypatch.setattr(output, "status", record_spinner)
            result = deployment_module.deploy(checkpoint="[red]bad[/red]")

        assert captured["checkpoint"] == "[red]bad[/red]"
        assert captured["workspace_id"] == WORKSPACE_ID
        assert captured["message"] == 'Deploying checkpoint "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)

    def test_deploy_failed_result_does_not_force_message_green(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, workspace_id, credentials=None):
                assert workspace_id == WORKSPACE_ID
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="failed",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.deploy(checkpoint="qwen3-run1-step-100")

        assert isinstance(result, OperationResult)
        assert result.status == "failed"
        assert result.message == "Deployment qwen3-run1-step-100 failed"
        assert result.exit_code == 1


@pytest.mark.usefixtures("mock_workspace_context")
class TestUndeploy:
    def test_undeploy_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def undeploy_checkpoint(
                self, checkpoint, *, workspace_id, credentials=None
            ):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return DeploymentSummary(
                    id=checkpoint_id,
                    checkpoint_name="qwen3-run1-step-100",
                    status="inactive",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.undeploy(checkpoint=checkpoint_id)
        assert captured == {
            "checkpoint": checkpoint_id,
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "undeploy"
        assert result.status == "success"
        assert result.resource == {
            "id": checkpoint_id,
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "inactive",
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }
        assert result.message == "Deployment qwen3-run1-step-100 inactive"

    def test_undeploy_escapes_checkpoint_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def undeploy_checkpoint(
                self, checkpoint, *, workspace_id, credentials=None
            ):
                captured["checkpoint"] = checkpoint
                captured["workspace_id"] = workspace_id
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="inactive",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        with override_output_context(
            format=OutputFormat.rich, interactive=True
        ) as output:
            monkeypatch.setattr(output, "status", record_spinner)
            result = deployment_module.undeploy(checkpoint="[red]bad[/red]")

        assert captured["checkpoint"] == "[red]bad[/red]"
        assert captured["workspace_id"] == WORKSPACE_ID
        assert captured["message"] == 'Undeploying checkpoint "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)

    def test_undeploy_failed_result_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def undeploy_checkpoint(
                self, checkpoint, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="failed",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.undeploy(checkpoint="qwen3-run1-step-100")

        assert isinstance(result, OperationResult)
        assert result.status == "failed"
        assert result.exit_code == 1


@pytest.mark.usefixtures("mock_workspace_context")
class TestRename:
    def test_rename_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def rename_checkpoint(
                self, checkpoint, new_name, *, workspace_id, credentials=None
            ):
                captured["checkpoint"] = checkpoint
                captured["new_name"] = new_name
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return RenameDeploymentResult(
                    id="dep_1",
                    old_checkpoint_name="old-name",
                    checkpoint_name="new-name",
                    status="active",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.rename(checkpoint="old-name", new_name="new-name")
        assert captured == {
            "checkpoint": "old-name",
            "new_name": "new-name",
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "deployment.rename"
        assert result.status == "success"
        assert result.resource == {
            "id": "dep_1",
            "old_checkpoint_name": "old-name",
            "checkpoint_name": "new-name",
            "status": "active",
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }
        assert result.message == "Renamed old-name -> new-name"

    def test_rename_warns_when_reregistration_fails(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def rename_checkpoint(
                self, checkpoint, new_name, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                return RenameDeploymentResult(
                    id="dep_1",
                    old_checkpoint_name="old-name",
                    checkpoint_name="new-name",
                    status="failed",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.rename(checkpoint="old-name", new_name="new-name")

        assert isinstance(result, OperationResult)
        assert result.status == "failed"
        assert result.display_next_steps == [
            "Warning: inference re-registration failed; deployment is marked as failed."
        ]
        assert result.exit_code == 1


@pytest.mark.usefixtures("mock_workspace_context")
class TestDelete:
    def test_delete_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def delete_deployment(self, checkpoint, *, workspace_id, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["workspace_id"] = workspace_id
                return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.delete(checkpoint="qwen3-run1-step-100", yes=True)
        assert captured == {
            "checkpoint": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
            "workspace_id": WORKSPACE_ID,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "deployment.delete"
        assert result.status == "success"
        assert result.resource == {
            "checkpoint": "qwen3-run1-step-100",
            "workspace": {"id": WORKSPACE_ID, "name": WORKSPACE_NAME},
            "project_root": str(PROJECT_ROOT),
        }
        assert result.message == 'Deployment for "qwen3-run1-step-100" deleted.'
