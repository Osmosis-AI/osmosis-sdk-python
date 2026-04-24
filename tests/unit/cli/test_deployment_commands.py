"""Tests for osmosis_ai.cli.commands.deployment."""

from __future__ import annotations

from io import StringIO

import pytest

import osmosis_ai.cli.commands.deployment as deployment_module
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import (
    DeploymentInfo,
    DeploymentSummary,
    PaginatedDeployments,
    RenameDeploymentResult,
)

AUTH_CREDENTIALS = object()


@pytest.fixture()
def console_capture(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    monkeypatch.setattr(deployment_module, "console", console)
    monkeypatch.setattr(utils_module, "console", console)
    return output


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.utils._require_auth",
        lambda: ("ws-test", AUTH_CREDENTIALS),
    )


class TestListDeployments:
    def test_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def list_deployments(self, limit=30, offset=0, *, credentials=None):
                return PaginatedDeployments(
                    deployments=[], total_count=0, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.list_deployments(limit=30, all_=False)
        assert "No deployments found" in console_capture.getvalue()

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
            def list_deployments(self, limit=30, offset=0, *, credentials=None):
                captured["limit"] = limit
                captured["offset"] = offset
                captured["credentials"] = credentials
                return PaginatedDeployments(
                    deployments=[dep], total_count=1, has_more=False
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.list_deployments(limit=10, all_=False)
        out = console_capture.getvalue()
        assert captured == {
            "limit": 10,
            "offset": 0,
            "credentials": AUTH_CREDENTIALS,
        }
        assert "Deployments" in out
        assert "qwen3-run1-step-100" in out
        assert "Qwen/Qwen3" in out


class TestInfo:
    def test_info_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def get_deployment(self, checkpoint, *, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
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
        deployment_module.info(checkpoint=checkpoint_id)
        out = console_capture.getvalue()
        assert captured == {
            "checkpoint": checkpoint_id,
            "credentials": AUTH_CREDENTIALS,
        }
        assert "qwen3-run1-step-100" in out
        assert "Qwen/Qwen3" in out
        assert "qwen3-run1" in out


class TestDeploy:
    def test_deploy_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def deploy_checkpoint(self, checkpoint, *, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                return DeploymentSummary(
                    id="dep_1",
                    checkpoint_name="qwen3-run1-step-100",
                    status="active",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.deploy(checkpoint="qwen3-run1-step-100")
        assert captured == {
            "checkpoint": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
        }
        assert "qwen3-run1-step-100" in console_capture.getvalue()


class TestUndeploy:
    def test_undeploy_accepts_checkpoint_uuid(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        checkpoint_id = "550e8400-e29b-41d4-a716-446655440000"
        captured: dict[str, object] = {}

        class FakeClient:
            def undeploy_checkpoint(self, checkpoint, *, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                return DeploymentSummary(
                    id=checkpoint_id,
                    checkpoint_name="qwen3-run1-step-100",
                    status="inactive",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.undeploy(checkpoint=checkpoint_id)
        out = console_capture.getvalue()
        assert captured == {
            "checkpoint": checkpoint_id,
            "credentials": AUTH_CREDENTIALS,
        }
        assert "qwen3-run1-step-100" in out
        assert "inactive" in out


class TestRename:
    def test_rename_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def rename_checkpoint(self, checkpoint, new_name, *, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["new_name"] = new_name
                captured["credentials"] = credentials
                return RenameDeploymentResult(
                    id="dep_1",
                    old_checkpoint_name="old-name",
                    checkpoint_name="new-name",
                    status="active",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.rename(checkpoint="old-name", new_name="new-name")
        assert captured == {
            "checkpoint": "old-name",
            "new_name": "new-name",
            "credentials": AUTH_CREDENTIALS,
        }
        assert "Renamed old-name -> new-name" in console_capture.getvalue()

    def test_rename_warns_when_reregistration_fails(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        class FakeClient:
            def rename_checkpoint(self, checkpoint, new_name, *, credentials=None):
                return RenameDeploymentResult(
                    id="dep_1",
                    old_checkpoint_name="old-name",
                    checkpoint_name="new-name",
                    status="failed",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.rename(checkpoint="old-name", new_name="new-name")
        assert "re-registration failed" in console_capture.getvalue()


class TestDelete:
    def test_delete_accepts_checkpoint_name(
        self, monkeypatch: pytest.MonkeyPatch, console_capture: StringIO
    ) -> None:
        captured: dict[str, object] = {}

        class FakeClient:
            def delete_deployment(self, checkpoint, *, credentials=None):
                captured["checkpoint"] = checkpoint
                captured["credentials"] = credentials
                return True

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        deployment_module.delete(checkpoint="qwen3-run1-step-100", yes=True)
        assert captured == {
            "checkpoint": "qwen3-run1-step-100",
            "credentials": AUTH_CREDENTIALS,
        }
        assert "deleted" in console_capture.getvalue()
