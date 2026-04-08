"""Tests for osmosis_ai.platform.api.client.OsmosisClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.platform.api.client import OsmosisClient, _safe_path


class TestCompleteUploadValidation:
    """Tests for parameter validation in OsmosisClient.complete_upload."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_no_parts_ok(self, mock_request: MagicMock) -> None:
        """Verify no error when parts is None (simple upload)."""
        mock_request.return_value = {
            "id": "file-1",
            "file_name": "file.jsonl",
            "file_size": 1024,
            "status": "uploaded",
        }
        client = OsmosisClient()
        result = client.complete_upload(file_id="file-1", parts=None)
        assert result.id == "file-1"
        assert result.status == "uploaded"
        mock_request.assert_called_once()
        # Simple upload uses short timeout
        call_kwargs = mock_request.call_args
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 30.0

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_with_parts_ok(self, mock_request: MagicMock) -> None:
        """Verify no error when parts are provided (multipart)."""
        mock_request.return_value = {
            "id": "file-2",
            "file_name": "large.jsonl",
            "file_size": 52428800,
            "status": "uploaded",
        }
        parts = [
            {"PartNumber": 1, "ETag": "etag-aaa"},
            {"PartNumber": 2, "ETag": "etag-bbb"},
        ]
        client = OsmosisClient()
        result = client.complete_upload(file_id="file-2", parts=parts)
        assert result.id == "file-2"
        assert result.status == "uploaded"
        # Verify parts payload was passed correctly
        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert payload["parts"] == parts
        assert "s3_key" not in payload
        assert "upload_id" not in payload
        # Verify multipart timeout (120s)
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 120.0

    def test_duplicate_part_numbers_raises(self) -> None:
        """Verify ValueError when duplicate part numbers are provided."""
        client = OsmosisClient()
        parts = [
            {"PartNumber": 1, "ETag": "etag-1"},
            {"PartNumber": 1, "ETag": "etag-2"},
        ]
        with pytest.raises(ValueError, match="Duplicate part numbers"):
            client.complete_upload(file_id="file-1", parts=parts)


class TestAbortUpload:
    """Tests for OsmosisClient.abort_upload."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_sends_empty_body(self, mock_request: MagicMock) -> None:
        """Verify abort sends empty body (server reads upload_id from DB)."""
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="file-1")
        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert payload == {}


class TestSafePath:
    """Tests for the _safe_path helper used to prevent path traversal."""

    def test_normal_id_unchanged(self) -> None:
        assert _safe_path("abc-123") == "abc-123"

    def test_uuid_unchanged(self) -> None:
        assert (
            _safe_path("550e8400-e29b-41d4-a716-446655440000")
            == "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_slash_encoded(self) -> None:
        assert _safe_path("../etc/passwd") == "..%2Fetc%2Fpasswd"

    def test_dot_dot_slash_encoded(self) -> None:
        result = _safe_path("../../secret")
        assert "/" not in result
        assert result == "..%2F..%2Fsecret"

    def test_space_encoded(self) -> None:
        assert _safe_path("my file") == "my%20file"


class TestURLPathSafety:
    """Verify that OsmosisClient methods apply _safe_path to URL path segments."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_complete_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "id": "f1",
            "file_name": "f.jsonl",
            "file_size": 100,
            "status": "uploaded",
        }
        client = OsmosisClient()
        client.complete_upload(file_id="a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/complete"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/abort"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_dataset_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "id": "f1",
            "file_name": "f.jsonl",
            "file_size": 100,
            "status": "uploaded",
        }
        client = OsmosisClient()
        client.get_dataset("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_delete_dataset_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.delete_dataset("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_training_run_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run": {"id": "r1", "status": "running"},
        }
        client = OsmosisClient()
        client.get_training_run("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb"


class TestGetModelAffectedResources:
    """Tests for OsmosisClient.get_model_affected_resources."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_base_model_with_runs(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_runs_using_model": [
                {"id": "r1", "training_run_name": "Run 1"},
            ],
            "creator_training_run": None,
        }
        client = OsmosisClient()
        result = client.get_model_affected_resources("m1", "base")
        assert len(result.training_runs_using_model) == 1
        assert result.has_blocking_runs is True
        path = mock_request.call_args[0][0]
        assert "/api/cli/models/m1/affected-resources" in path
        assert "type=base" in path

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_output_model_with_creator(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_runs_using_model": [],
            "creator_training_run": {
                "id": "r2",
                "training_run_name": "Creator Run",
            },
        }
        client = OsmosisClient()
        result = client.get_model_affected_resources("m2", "output")
        assert result.creator_training_run is not None
        assert result.creator_training_run.training_run_name == "Creator Run"
        assert result.has_blocking_runs is False

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_path_encodes_model_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_runs_using_model": [],
            "creator_training_run": None,
        }
        client = OsmosisClient()
        client.get_model_affected_resources("a/b")
        path = mock_request.call_args[0][0]
        assert "a%2Fb" in path


class TestGetWorkspaceDeletionStatus:
    """Tests for OsmosisClient.get_workspace_deletion_status."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_can_delete(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "can_delete": True,
            "is_owner": True,
            "is_last_workspace": False,
            "has_running_processes": False,
            "feature_pipelines": {"count": 0, "valid": True},
            "training_runs": {"count": 0, "valid": True},
            "models": {"count": 0, "valid": True},
        }
        client = OsmosisClient()
        result = client.get_workspace_deletion_status("ws-1")
        assert result.can_delete is True
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/workspaces/ws-1/deletion-status"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_with_running_processes(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "can_delete": False,
            "is_owner": True,
            "is_last_workspace": False,
            "has_running_processes": True,
            "feature_pipelines": {"count": 1, "valid": False},
            "training_runs": {"count": 0, "valid": True},
            "models": {"count": 0, "valid": True},
        }
        client = OsmosisClient()
        result = client.get_workspace_deletion_status("ws-2")
        assert result.can_delete is False
        assert result.has_running_processes is True

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_require_workspace_false(self, mock_request: MagicMock) -> None:
        """Verify deletion-status does not require workspace header."""
        mock_request.return_value = {
            "can_delete": True,
            "is_owner": True,
            "is_last_workspace": False,
            "has_running_processes": False,
            "feature_pipelines": {"count": 0, "valid": True},
            "training_runs": {"count": 0, "valid": True},
            "models": {"count": 0, "valid": True},
        }
        client = OsmosisClient()
        client.get_workspace_deletion_status("ws-1")
        call_kwargs = mock_request.call_args
        assert call_kwargs.kwargs.get("require_workspace") is False


class TestGetTrainingRunMetrics:
    """Tests for OsmosisClient.get_training_run_metrics."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_returns_parsed_metrics(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run_id": "run-1",
            "status": "finished",
            "overview": {
                "mlflow_run_id": "mlflow-1",
                "mlflow_status": "FINISHED",
                "duration_ms": 3600000,
                "duration_formatted": "1h",
                "reward": 0.85,
                "reward_increase_delta": 0.15,
                "examples_processed_count": 5000,
            },
            "metrics": [
                {
                    "metric_key": "rollout/raw_reward",
                    "title": "Training Reward",
                    "data_points": [
                        {"step": 0, "value": 0.5, "timestamp": 1711800000000},
                    ],
                },
            ],
        }
        client = OsmosisClient()
        result = client.get_training_run_metrics("run-1")
        assert result.training_run_id == "run-1"
        assert result.overview.reward == 0.85
        assert len(result.metrics) == 1
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/run-1/metrics"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_encodes_run_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run_id": "a/b",
            "status": "finished",
            "overview": {
                "mlflow_run_id": "m",
                "mlflow_status": "FINISHED",
                "duration_ms": None,
                "duration_formatted": None,
                "reward": None,
                "reward_increase_delta": None,
                "examples_processed_count": None,
            },
            "metrics": [],
        }
        client = OsmosisClient()
        client.get_training_run_metrics("a/b")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb/metrics"
