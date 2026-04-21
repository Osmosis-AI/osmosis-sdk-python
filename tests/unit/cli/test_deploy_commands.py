"""Tests for osmosis deploy {create, list, status, delete, rename}."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.deploy as deploy_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    CreateDeploymentResult,
    DeploymentInfo,
    LoraCheckpointInfo,
    PaginatedDeployments,
    RenameDeploymentResult,
    TrainingRunCheckpoints,
)


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(deploy_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", object()),
    )


def _make_ckpts(steps: list[int]) -> TrainingRunCheckpoints:
    return TrainingRunCheckpoints(
        training_run_id="run_1",
        training_run_name="qwen3-run1",
        checkpoints=[
            LoraCheckpointInfo(
                id=f"cp_{s}",
                checkpoint_step=s,
                status="uploaded",
                created_at="2026-04-20T00:00:00Z",
            )
            for s in steps
        ],
    )


class TestCreate:
    def test_happy_path_with_step(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = {}

        class FakeClient:
            def list_training_run_checkpoints(self, name, *, credentials=None):
                return _make_ckpts([100, 200])

            def create_deployment(
                self,
                *,
                training_run,
                checkpoint_step=None,
                lora_checkpoint_id=None,
                lora_name=None,
                credentials=None,
            ):
                captured["training_run"] = training_run
                captured["checkpoint_step"] = checkpoint_step
                captured["lora_name"] = lora_name
                return CreateDeploymentResult(
                    id="dep_1",
                    lora_name=lora_name or "qwen3-run1-step-100-lora",
                    status="deployed",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.create(training_run="qwen3-run1", step=100, name=None, yes=True)
        assert captured["training_run"] == "qwen3-run1"
        assert captured["checkpoint_step"] == 100
        assert captured["lora_name"] is None  # server generates
        assert "Deployment created" in console_capture.getvalue()

    def test_default_step_picks_latest(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = {}

        class FakeClient:
            def list_training_run_checkpoints(self, name, *, credentials=None):
                # Intentionally unsorted to verify defensive sort
                return _make_ckpts([50, 200, 100])

            def create_deployment(self, *, training_run, **kw):
                captured.update(kw)
                return CreateDeploymentResult(
                    id="dep_1", lora_name="auto", status="deployed"
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.create(training_run="qwen3-run1", step=None, name=None, yes=True)
        assert captured["checkpoint_step"] == 200

    def test_no_checkpoints_errors(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.cli.errors import CLIError

        class FakeClient:
            def list_training_run_checkpoints(self, name, *, credentials=None):
                return TrainingRunCheckpoints(
                    training_run_id="run_1",
                    training_run_name="qwen3-run1",
                    checkpoints=[],
                )

            def create_deployment(self, *a, **kw):  # pragma: no cover
                raise AssertionError("should not create")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        with pytest.raises(CLIError, match="No uploaded checkpoints"):
            deploy_module.create(
                training_run="qwen3-run1", step=None, name=None, yes=True
            )

    def test_unknown_step_errors_with_available(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        from osmosis_ai.cli.errors import CLIError

        class FakeClient:
            def list_training_run_checkpoints(self, name, *, credentials=None):
                return _make_ckpts([100, 200])

            def create_deployment(self, *a, **kw):  # pragma: no cover
                raise AssertionError("should not create")

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        with pytest.raises(CLIError) as excinfo:
            deploy_module.create(
                training_run="qwen3-run1", step=999, name=None, yes=True
            )
        assert "100" in str(excinfo.value)
        assert "200" in str(excinfo.value)

    def test_explicit_name_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = {}

        class FakeClient:
            def list_training_run_checkpoints(self, name, *, credentials=None):
                return _make_ckpts([100])

            def create_deployment(self, *, training_run, **kw):
                captured["lora_name"] = kw.get("lora_name")
                return CreateDeploymentResult(
                    id="dep_1", lora_name="my-custom", status="deployed"
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.create(
            training_run="qwen3-run1",
            step=100,
            name="my-custom",
            yes=True,
        )
        assert captured["lora_name"] == "my-custom"


class TestList:
    def test_empty(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_deployments(self, limit=30, offset=0, *, credentials=None):
                return PaginatedDeployments(
                    deployments=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.list_deployments(limit=30, all_=False)
        assert "No deployments found" in console_capture.getvalue()

    def test_with_deployments(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        dep = DeploymentInfo(
            id="dep_1",
            lora_name="qwen3-run1-step-100-lora",
            status="deployed",
            base_model="Qwen/Qwen3-30B",
            checkpoint_step=100,
            training_run_id="run_1",
            training_run_name="qwen3-run1",
            created_at="2026-04-20T00:00:00Z",
        )

        class FakeClient:
            def list_deployments(self, limit=30, offset=0, *, credentials=None):
                return PaginatedDeployments(
                    deployments=[dep], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.list_deployments(limit=30, all_=False)
        out = console_capture.getvalue()
        assert "Deployments" in out
        assert "qwen3-run1-step-100-lora" in out

    def test_has_more_truncation(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        dep = DeploymentInfo(
            id="dep_1",
            lora_name="x",
            status="deployed",
            base_model="Qwen/Qwen3",
            checkpoint_step=1,
        )

        class FakeClient:
            def list_deployments(self, limit=1, offset=0, *, credentials=None):
                return PaginatedDeployments(
                    deployments=[dep], total_count=5, has_more=True
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.list_deployments(limit=1, all_=False)
        out = console_capture.getvalue()
        assert "Showing 1 of 5 deployments" in out
        assert "--all" in out


class TestStatus:
    def test_full_details(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        dep = DeploymentInfo(
            id="dep_1",
            lora_name="my-lora",
            status="deployed",
            base_model="Qwen/Qwen3-30B",
            checkpoint_step=100,
            training_run_id="run_1",
            training_run_name="qwen3-run1",
            creator_name="brian",
            created_at="2026-04-20T00:00:00Z",
        )

        class FakeClient:
            def get_deployment(self, name, *, credentials=None):
                return dep

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.status(name="my-lora")
        out = console_capture.getvalue()
        assert "my-lora" in out
        assert "Qwen/Qwen3-30B" in out
        assert "step 100" in out
        assert "qwen3-run1" in out
        assert "brian" in out

    def test_minimal_fields(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        dep = DeploymentInfo(
            id="dep_1",
            lora_name="my-lora",
            status="deployed",
            base_model="",
            checkpoint_step=0,
        )

        class FakeClient:
            def get_deployment(self, name, *, credentials=None):
                return dep

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.status(name="my-lora")
        out = console_capture.getvalue()
        assert "my-lora" in out
        assert "Deployment" in out


class TestRename:
    def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured = {}

        class FakeClient:
            def rename_deployment(self, name, new_name, *, credentials=None):
                captured["old"] = name
                captured["new"] = new_name
                return RenameDeploymentResult(id="dep_1", lora_name=new_name)

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.rename(name="foo", new_name="bar", yes=True)
        assert captured == {"old": "foo", "new": "bar"}
        assert "Renamed" in console_capture.getvalue()


class TestDelete:
    def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def delete_deployment(self, name, *, credentials=None):
                captured["deleted"] = name
                return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deploy_module.delete(name="my-lora", yes=True)
        assert captured["deleted"] == "my-lora"
        assert 'Deployment "my-lora" deleted' in console_capture.getvalue()
