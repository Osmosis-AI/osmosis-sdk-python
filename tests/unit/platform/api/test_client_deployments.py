"""Tests for OsmosisClient deployment + checkpoint methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
    def test_with_search(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployments": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        client.list_deployments(search="my-lora")
        args, _ = mock_req.call_args
        assert "search=my-lora" in args[0]

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_no_search_no_param(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployments": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        client.list_deployments()
        args, _ = mock_req.call_args
        assert "search=" not in args[0]


class TestCreateDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_with_step(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "dep_1",
                "lora_name": "auto-generated",
                "status": "deployed",
            }
        }
        client = OsmosisClient()
        result = client.create_deployment(
            training_run="qwen3-run1", checkpoint_step=100
        )
        assert result.id == "dep_1"

        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {
            "training_run": "qwen3-run1",
            "checkpoint_step": 100,
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_with_name_override(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "dep_1",
                "lora_name": "custom",
                "status": "deployed",
            }
        }
        client = OsmosisClient()
        client.create_deployment(
            training_run="qwen3-run1", checkpoint_step=100, lora_name="custom"
        )
        _, kwargs = mock_req.call_args
        assert kwargs["data"]["lora_name"] == "custom"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_none_fields_omitted(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {"id": "dep_1", "lora_name": "x", "status": "deployed"}
        }
        client = OsmosisClient()
        client.create_deployment(training_run="run", checkpoint_step=0)
        _, kwargs = mock_req.call_args
        assert set(kwargs["data"].keys()) == {"training_run", "checkpoint_step"}

    def test_requires_checkpoint_step(self) -> None:
        client = OsmosisClient()
        with pytest.raises(ValueError, match="checkpoint_step is required"):
            client.create_deployment(
                training_run="run",
                checkpoint_step=None,  # type: ignore[arg-type]
            )


class TestGetDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_by_name(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "dep_1",
                "lora_name": "x",
                "status": "deployed",
                "base_model": "Qwen/Qwen3",
                "checkpoint_step": 1,
            }
        }
        client = OsmosisClient()
        result = client.get_deployment("my-lora")
        assert result.lora_name == "x"
        args, _ = mock_req.call_args
        assert args[0] == "/api/cli/deployments/my-lora"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "deployment": {
                "id": "d",
                "lora_name": "x",
                "status": "deployed",
                "base_model": "q",
                "checkpoint_step": 0,
            }
        }
        client = OsmosisClient()
        client.get_deployment("../bad")
        args, _ = mock_req.call_args
        assert "../" not in args[0]


class TestDeleteDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_delete(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"deleted": True}
        client = OsmosisClient()
        assert client.delete_deployment("my-lora") is True
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/my-lora"
        assert kwargs["method"] == "DELETE"


class TestRenameDeployment:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_rename(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"deployment": {"id": "dep_1", "lora_name": "new-name"}}
        client = OsmosisClient()
        result = client.rename_deployment("old-name", "new-name")
        assert result.lora_name == "new-name"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/deployments/old-name"
        assert kwargs["method"] == "PATCH"
        assert kwargs["data"] == {"lora_name": "new-name"}


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
