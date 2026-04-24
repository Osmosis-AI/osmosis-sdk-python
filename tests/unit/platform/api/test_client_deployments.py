"""Tests for OsmosisClient deployment + checkpoint methods.

Covers the checkpoint-centric API:
    list_deployments / get_deployment / deploy_checkpoint /
    undeploy_checkpoint / rename_checkpoint / delete_deployment /
    list_training_run_checkpoints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient


class TestListDeployments:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_basic(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployments": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        result = client.list_deployments(limit=10, offset=5)
        assert result.total_count == 0
        args, _kwargs = mock_req.call_args
        assert "/api/cli/deployments?" in args[0]
        assert "limit=10" in args[0]
        assert "offset=5" in args[0]

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_defaults(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployments": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        client.list_deployments()
        args, _ = mock_req.call_args
        assert "/api/cli/deployments?" in args[0]
        assert "limit=" in args[0]
        assert "offset=0" in args[0]


class TestGetDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_by_checkpoint_name(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "dep_1",
                "checkpoint_name": "qwen3-step-100",
                "status": "active",
                "checkpoint_step": 100,
                "base_model": "Qwen/Qwen3",
            }
        }
        client = OsmosisClient()
        result = client.get_deployment("qwen3-step-100")
        assert result.checkpoint_name == "qwen3-step-100"
        args, _ = mock_req.call_args
        assert args[0] == "/api/cli/deployments/qwen3-step-100"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "d",
                "checkpoint_name": "x",
                "status": "active",
                "checkpoint_step": 0,
                "base_model": "q",
            }
        }
        client = OsmosisClient()
        client.get_deployment("../bad")
        args, _ = mock_req.call_args
        assert "../" not in args[0]


class TestDeployCheckpoint:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_deploy(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "dep_1",
                "checkpoint_name": "qwen3-step-100",
                "status": "active",
            }
        }
        client = OsmosisClient()
        result = client.deploy_checkpoint("qwen3-step-100")
        assert result.id == "dep_1"
        assert result.status == "active"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/qwen3-step-100/deploy"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {}

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_deploy_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {"id": "d", "checkpoint_name": "x", "status": "active"}
        }
        client = OsmosisClient()
        client.deploy_checkpoint("../bad")
        args, _ = mock_req.call_args
        assert "../" not in args[0]


class TestUndeployCheckpoint:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_undeploy(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "id": "dep_1",
            "checkpoint_name": "qwen3-step-100",
            "status": "inactive",
        }
        client = OsmosisClient()
        result = client.undeploy_checkpoint("qwen3-step-100")
        assert result.status == "inactive"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/qwen3-step-100/undeploy"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {}


class TestRenameCheckpoint:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_rename(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "id": "dep_1",
            "old_checkpoint_name": "old-name",
            "checkpoint_name": "new-name",
            "status": "active",
        }
        client = OsmosisClient()
        result = client.rename_checkpoint("old-name", "new-name")
        assert result.checkpoint_name == "new-name"
        assert result.old_checkpoint_name == "old-name"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/old-name"
        assert kwargs["method"] == "PATCH"
        assert kwargs["data"] == {"checkpoint_name": "new-name"}


class TestDeleteDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_delete(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"deleted": True}
        client = OsmosisClient()
        assert client.delete_deployment("qwen3-step-100") is True
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/qwen3-step-100"
        assert kwargs["method"] == "DELETE"


class TestListTrainingRunCheckpoints:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "training_run_id": "run_1",
            "training_run_name": "qwen3-run1",
            "checkpoints": [
                {
                    "id": "cp_1",
                    "checkpoint_step": 100,
                    "status": "uploaded",
                    "created_at": "2026-04-20T00:00:00Z",
                }
            ],
        }
        client = OsmosisClient()
        result = client.list_training_run_checkpoints("qwen3-run1")
        assert len(result.checkpoints) == 1
        args, _ = mock_req.call_args
        assert args[0] == "/api/cli/training-runs/qwen3-run1/checkpoints"
