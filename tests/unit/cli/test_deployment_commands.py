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
    TrainingRun,
    TrainingRunCheckpoints,
)

AUTH_CREDENTIALS = object()
GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
PROJECT_ROOT = Path("/repo")


def assert_git_context(data: dict[str, object]) -> None:
    assert data["workspace_directory"] == "/repo"
    assert data["git"] == {
        "identity": GIT_IDENTITY,
        "remote_url": REPO_URL,
    }
    assert "workspace" not in data


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(deployment_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture()
def mock_git_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    context = SimpleNamespace(
        workspace_directory=PROJECT_ROOT,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=AUTH_CREDENTIALS,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils.require_git_workspace_directory_context",
        lambda: context,
    )
    return context


@pytest.fixture()
def linked_project(mock_git_context: None) -> None:
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
        lambda self, checkpoint, *, credentials, git_identity: _deployment_summary(
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
    assert "Osmosis workspace directory" in capsys.readouterr().err


class TestDeployWizardHelper:
    def test_select_checkpoint_returns_selected_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, mock_git_context: SimpleNamespace
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
            def list_training_runs(self, *, git_identity, credentials=None):
                calls.append(("runs", git_identity))
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                calls.append(("checkpoints", git_identity))
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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

        selected = deployment_module._select_checkpoint_for_deploy(mock_git_context)

        assert selected == "reward-run-step-100"
        assert calls == [("runs", GIT_IDENTITY), ("checkpoints", GIT_IDENTITY)]

    def test_select_checkpoint_cancel_at_run(
        self, monkeypatch: pytest.MonkeyPatch, mock_git_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")

        class FakeClient:
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                raise AssertionError("checkpoint list should not be fetched")

        def fake_select_list(message, *, items, actions):
            assert message == "Choose a training run"
            assert items
            assert actions[0].title == "Cancel"
            return actions[0].value

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        assert deployment_module._select_checkpoint_for_deploy(mock_git_context) is None

    def test_select_checkpoint_cancel_at_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, mock_git_context: SimpleNamespace
    ) -> None:
        run = TrainingRun(id="run_1", name="reward-run", status="finished")
        checkpoint = LoraCheckpointInfo(
            id="cp_1",
            checkpoint_name="reward-run-step-100",
            checkpoint_step=100,
            status="ready",
        )

        class FakeClient:
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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
                assert actions[-1].title == "Cancel"
                return actions[-1].value
            raise AssertionError(f"unexpected prompt: {message}")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        assert deployment_module._select_checkpoint_for_deploy(mock_git_context) is None

    def test_select_checkpoint_back_returns_to_run_selection(
        self, monkeypatch: pytest.MonkeyPatch, mock_git_context: SimpleNamespace
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
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[first_run, second_run],
                    total_count=2,
                    has_more=False,
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                checkpoint_calls.append(run_id)
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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

        selected = deployment_module._select_checkpoint_for_deploy(mock_git_context)

        assert selected == "second-run-step-100"
        assert checkpoint_calls == ["run_1", "run_2"]

    def test_select_checkpoint_no_training_runs_raises_before_prompt(
        self, monkeypatch: pytest.MonkeyPatch, mock_git_context: SimpleNamespace
    ) -> None:
        class FakeClient:
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[], total_count=0, has_more=False
                )

        def fake_select_list(message, *, items, actions):
            raise AssertionError("select_list should not be called")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)

        with pytest.raises(CLIError, match="No training runs"):
            deployment_module._select_checkpoint_for_deploy(mock_git_context)

    def test_select_checkpoint_no_checkpoints_confirms_return_to_run_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_git_context: SimpleNamespace,
        console_capture: StringIO,
    ) -> None:
        first_run = TrainingRun(id="run_1", name="empty-run", status="finished")
        second_run = TrainingRun(id="run_2", name="reward-run", status="finished")
        checkpoint = LoraCheckpointInfo(
            id="cp_2",
            checkpoint_name="reward-run-step-100",
            checkpoint_step=100,
            status="ready",
        )
        selected_runs = iter([first_run, second_run])
        checkpoint_calls: list[str] = []
        confirm_messages: list[str] = []

        class FakeClient:
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[first_run, second_run],
                    total_count=2,
                    has_more=False,
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                checkpoint_calls.append(run_id)
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                checkpoints = [] if run_id == "run_1" else [checkpoint]
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name=run_id,
                    checkpoints=checkpoints,
                )

        def fake_select_list(message, *, items, actions):
            assert items
            if message == "Choose a training run":
                return next(selected_runs)
            if message == "Choose a checkpoint":
                return checkpoint
            raise AssertionError(f"unexpected prompt: {message}")

        def fake_confirm(message, *, default=True):
            confirm_messages.append(message)
            assert default is True
            return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)
        monkeypatch.setattr(deployment_module, "confirm", fake_confirm)

        selected = deployment_module._select_checkpoint_for_deploy(mock_git_context)

        assert selected == "reward-run-step-100"
        assert checkpoint_calls == ["run_1", "run_2"]
        assert confirm_messages == ["Choose another training run?"]
        output = console_capture.getvalue()
        assert 'No deployable checkpoints found for training run "empty-run".' in output

    def test_select_checkpoint_no_checkpoints_decline_cancels_deploy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_git_context: SimpleNamespace,
        console_capture: StringIO,
    ) -> None:
        run = TrainingRun(id="run_1", name="empty-run", status="finished")

        class FakeClient:
            def list_training_runs(self, *, git_identity, credentials=None):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return PaginatedTrainingRuns(
                    training_runs=[run], total_count=1, has_more=False
                )

            def list_training_run_checkpoints(
                self, run_id, *, git_identity, credentials=None
            ):
                assert run_id == "run_1"
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
                return TrainingRunCheckpoints(
                    training_run_id=run_id,
                    training_run_name="empty-run",
                    checkpoints=[],
                )

        def fake_select_list(message, *, items, actions):
            assert message == "Choose a training run"
            assert items
            return run

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(deployment_module, "select_list", fake_select_list)
        monkeypatch.setattr(deployment_module, "confirm", lambda _message: False)

        assert deployment_module._select_checkpoint_for_deploy(mock_git_context) is None

        assert 'No deployable checkpoints found for training run "empty-run".' in (
            console_capture.getvalue()
        )


@pytest.mark.usefixtures("mock_git_context")
class TestListDeployments:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_deployments(
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                assert credentials is AUTH_CREDENTIALS
                assert git_identity == GIT_IDENTITY
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
        assert_git_context(result.extra)

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
                self, limit=30, offset=0, *, git_identity, credentials=None
            ):
                captured["limit"] = limit
                captured["offset"] = offset
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
                return PaginatedDeployments(
                    deployments=[dep], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        result = deployment_module.list_deployments(limit=10, all_=False)
        assert captured == {
            "limit": 10,
            "offset": 0,
            "credentials": AUTH_CREDENTIALS,
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, ListResult)
        assert result.title == "Deployments"
        assert result.items[0]["checkpoint_name"] == "qwen3-run1-step-100"
        assert result.items[0]["base_model"] == "Qwen/Qwen3"


@pytest.mark.usefixtures("mock_git_context")
class TestInfo:
    def test_info_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def get_deployment(self, checkpoint, *, git_identity, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
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
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, DetailResult)
        assert result.title == "Deployment"
        assert result.data["checkpoint_name"] == "qwen3-run1-step-100"
        assert result.data["base_model"] == "Qwen/Qwen3"
        assert result.data["training_run_name"] == "qwen3-run1"
        assert_git_context(result.data)

    def test_info_escapes_status_for_rich_table(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def get_deployment(self, checkpoint, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
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


@pytest.mark.usefixtures("mock_git_context")
class TestDeploy:
    def test_deploy_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
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
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "deploy"
        assert result.status == "success"
        assert result.resource == {
            "id": "dep_1",
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "active",
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
        }
        assert result.message == "Deployment qwen3-run1-step-100 active"

    def test_deploy_interactive_cancel_returns_cancelled_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
                raise AssertionError("checkpoint should not be deployed")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(
            deployment_module,
            "_select_checkpoint_for_deploy",
            lambda _workspace: None,
        )

        with override_output_context(format=OutputFormat.rich, interactive=True):
            result = deployment_module.deploy(checkpoint=None)

        assert isinstance(result, OperationResult)
        assert result.operation == "deploy"
        assert result.status == "cancelled"
        assert result.exit_code == 0
        assert result.message == "Deploy cancelled."
        assert result.resource == {
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
        }

    def test_deploy_escapes_checkpoint_in_spinner(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def record_spinner(message):
            captured["message"] = message
            yield

        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["git_identity"] = git_identity
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
        assert captured["git_identity"] == GIT_IDENTITY
        assert captured["message"] == 'Deploying checkpoint "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)

    def test_deploy_failed_result_does_not_force_message_green(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
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


@pytest.mark.usefixtures("mock_git_context")
class TestUndeploy:
    def test_undeploy_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def undeploy_checkpoint(
                self, checkpoint, *, git_identity, credentials=None
            ):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                captured["git_identity"] = git_identity
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
            "git_identity": GIT_IDENTITY,
        }
        assert isinstance(result, OperationResult)
        assert result.operation == "undeploy"
        assert result.status == "success"
        assert result.resource == {
            "id": checkpoint_id,
            "checkpoint_name": "qwen3-run1-step-100",
            "status": "inactive",
            "git": {"identity": GIT_IDENTITY, "remote_url": REPO_URL},
            "workspace_directory": "/repo",
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
                self, checkpoint, *, git_identity, credentials=None
            ):
                captured["checkpoint"] = checkpoint
                captured["git_identity"] = git_identity
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
        assert captured["git_identity"] == GIT_IDENTITY
        assert captured["message"] == 'Undeploying checkpoint "\\[red]bad\\[/red]"...'
        assert isinstance(result, OperationResult)

    def test_undeploy_failed_result_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def undeploy_checkpoint(
                self, checkpoint, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
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
