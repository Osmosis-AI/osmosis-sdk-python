"""Tests for OsmosisClient model + checkpoint methods.

Covers the model-centric API:
    list_base_models / list_lora_models / get_lora_model /
    deploy_lora_model / undeploy_lora_model / list_training_run_checkpoints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient

GIT_IDENTITY = "git_test"


class TestListBaseModels:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_basic(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "models": [
                {
                    "id": "model_1",
                    "model_name": "Qwen/Qwen3",
                    "base_model": "Qwen/Qwen3",
                    "creator_name": "brian",
                    "created_at": "2026-04-20T00:00:00Z",
                    "updated_at": "2026-04-20T00:00:00Z",
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        result = client.list_base_models(limit=10, offset=5, git_identity=GIT_IDENTITY)
        assert len(result.models) == 1
        assert result.models[0].model_name == "Qwen/Qwen3"
        assert result.total_count == 1
        assert result.has_more is False
        assert result.next_offset is None
        args, kwargs = mock_req.call_args
        assert "/api/cli/models/base?" in args[0]
        assert "limit=10" in args[0]
        assert "offset=5" in args[0]
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_defaults(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "models": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        result = client.list_base_models(git_identity=GIT_IDENTITY)
        assert result.models == []
        args, kwargs = mock_req.call_args
        assert "/api/cli/models/base?" in args[0]
        assert "limit=" in args[0]
        assert "offset=0" in args[0]
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs


class TestListLoraModels:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_basic(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "models": [
                {
                    "id": "lora_1",
                    "model_name": "qwen3-run1-step-100",
                    "base_model": "Qwen/Qwen3",
                    "training_run_name": "qwen3-run1",
                    "checkpoint_step": 100,
                    "deployment_status": "active",
                    "created_at": "2026-04-21T00:00:00Z",
                }
            ],
            "total_count": 3,
            "has_more": True,
            "next_offset": 1,
        }
        client = OsmosisClient()
        result = client.list_lora_models(limit=10, offset=5, git_identity=GIT_IDENTITY)
        assert len(result.models) == 1
        assert result.models[0].deployment_status == "active"
        assert result.total_count == 3
        assert result.has_more is True
        assert result.next_offset == 1
        args, kwargs = mock_req.call_args
        assert "/api/cli/models/lora?" in args[0]
        assert "limit=10" in args[0]
        assert "offset=5" in args[0]
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_defaults(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "models": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        client = OsmosisClient()
        result = client.list_lora_models(git_identity=GIT_IDENTITY)
        assert result.models == []
        args, kwargs = mock_req.call_args
        assert "/api/cli/models/lora?" in args[0]
        assert "limit=" in args[0]
        assert "offset=0" in args[0]
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs


class TestGetLoraModel:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "id": "lora_1",
            "model_name": "qwen3-run1-step-100",
            "base_model": "Qwen/Qwen3",
            "training_run_name": "qwen3-run1",
            "checkpoint_step": 100,
            "reward": 0.85,
            "deployment_status": "active",
            "created_at": "2026-04-21T00:00:00Z",
            "hf_upload_status": "uploaded",
            "hf_url": "https://huggingface.co/acme/qwen3-run1-step-100",
            "uploaded_by": "Ada Lovelace",
            "platform_url": "https://platform.osmosis.ai/acme/models/lora_1",
        }
        client = OsmosisClient()
        result = client.get_lora_model("qwen3-run1-step-100", git_identity=GIT_IDENTITY)
        assert result.id == "lora_1"
        assert result.model_name == "qwen3-run1-step-100"
        assert result.hf_upload_status == "uploaded"
        assert result.hf_url == "https://huggingface.co/acme/qwen3-run1-step-100"
        assert result.uploaded_by == "Ada Lovelace"
        assert result.has_deployment_info is True
        assert result.platform_url == "https://platform.osmosis.ai/acme/models/lora_1"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/models/qwen3-run1-step-100"
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_without_deployment_fields(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"id": "lora_1", "model_name": "qwen3-run1-step-100"}
        client = OsmosisClient()
        result = client.get_lora_model("qwen3-run1-step-100", git_identity=GIT_IDENTITY)
        assert result.has_deployment_info is False
        assert result.deployment_status is None

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"id": "l", "model_name": "x"}
        client = OsmosisClient()
        client.get_lora_model("../bad", git_identity=GIT_IDENTITY)
        args, _kwargs = mock_req.call_args
        assert "../" not in args[0]


class TestDeployLoraModel:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_deploy(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "id": "lora_1",
            "model_name": "qwen3-run1-step-100",
            "status": "active",
        }
        client = OsmosisClient()
        result = client.deploy_lora_model(
            "qwen3-run1-step-100", git_identity=GIT_IDENTITY
        )
        assert result.id == "lora_1"
        assert result.model_name == "qwen3-run1-step-100"
        assert result.status == "active"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/models/qwen3-run1-step-100/deploy"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {}
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_deploy_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"id": "l", "model_name": "x", "status": "active"}
        client = OsmosisClient()
        client.deploy_lora_model("../bad", git_identity=GIT_IDENTITY)
        args, kwargs = mock_req.call_args
        assert "../" not in args[0]
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs


class TestUndeployLoraModel:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_undeploy(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "id": "lora_1",
            "model_name": "qwen3-run1-step-100",
            "status": "inactive",
        }
        client = OsmosisClient()
        result = client.undeploy_lora_model(
            "qwen3-run1-step-100", git_identity=GIT_IDENTITY
        )
        assert result.status == "inactive"
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/models/qwen3-run1-step-100/undeploy"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {}
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_undeploy_urlencodes_path(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"id": "l", "model_name": "x", "status": "inactive"}
        client = OsmosisClient()
        client.undeploy_lora_model("../bad", git_identity=GIT_IDENTITY)
        args, _kwargs = mock_req.call_args
        assert "../" not in args[0]


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
        result = client.list_training_run_checkpoints(
            "qwen3-run1", git_identity=GIT_IDENTITY
        )
        assert len(result.checkpoints) == 1
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/training-runs/qwen3-run1/checkpoints"
        assert kwargs["git_identity"] == GIT_IDENTITY
        assert "workspace_id" not in kwargs
