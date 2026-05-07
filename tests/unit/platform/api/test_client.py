"""Tests for osmosis_ai.platform.api.client.OsmosisClient."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
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
        result = client.complete_upload(
            file_id="file-1", parts=None, workspace_id="ws_test"
        )
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
        result = client.complete_upload(
            file_id="file-2", parts=parts, workspace_id="ws_test"
        )
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
            client.complete_upload(
                file_id="file-1", parts=parts, workspace_id="ws_test"
            )


class TestAbortUpload:
    """Tests for OsmosisClient.abort_upload."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_sends_empty_body(self, mock_request: MagicMock) -> None:
        """Verify abort sends empty body (server reads upload_id from DB)."""
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="file-1", workspace_id="ws_test")
        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert payload == {}


class TestListDatasets:
    """Tests for OsmosisClient.list_datasets."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list_datasets_passes_workspace_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"datasets": [], "total_count": 0}

        OsmosisClient().list_datasets(credentials=object(), workspace_id="ws_123")

        assert mock_request.call_args.kwargs["workspace_id"] == "ws_123"


class TestWorkspaceIdPassthrough:
    """Tests representative workspace-scoped methods pass workspace_id."""

    @pytest.mark.parametrize(
        ("call_client", "response"),
        [
            (
                lambda client: client.create_dataset(
                    "data.jsonl", 100, "jsonl", workspace_id="ws_123"
                ),
                {
                    "id": "file-1",
                    "file_name": "data.jsonl",
                    "file_size": 100,
                    "status": "uploaded",
                },
            ),
            (
                lambda client: client.submit_training_run(
                    model_path="openai/gpt-oss",
                    dataset="dataset-1",
                    rollout_name="rollout",
                    entrypoint="rollout.py",
                    workspace_id="ws_123",
                ),
                {
                    "id": "run-1",
                    "name": "Run 1",
                    "status": "pending",
                    "created_at": "2026-05-03T00:00:00Z",
                },
            ),
            (
                lambda client: client.list_training_runs(workspace_id="ws_123"),
                {"training_runs": [], "total_count": 0},
            ),
            (
                lambda client: client.list_rollouts(workspace_id="ws_123"),
                {"rollouts": [], "total_count": 0},
            ),
            (
                lambda client: client.list_base_models(workspace_id="ws_123"),
                {"models": [], "total_count": 0},
            ),
            (
                lambda client: client.list_deployments(workspace_id="ws_123"),
                {"deployments": [], "total_count": 0},
            ),
            (
                lambda client: client.deploy_checkpoint(
                    "checkpoint-1", workspace_id="ws_123"
                ),
                {
                    "deployment": {
                        "id": "dep-1",
                        "checkpoint_name": "checkpoint-1",
                        "status": "active",
                    }
                },
            ),
            (
                lambda client: client.list_training_run_checkpoints(
                    "run-1", workspace_id="ws_123"
                ),
                {
                    "training_run_id": "run-1",
                    "training_run_name": "Run 1",
                    "checkpoints": [],
                },
            ),
        ],
    )
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_workspace_scoped_methods_pass_workspace_id(
        self,
        mock_request: MagicMock,
        call_client: Callable[[OsmosisClient], object],
        response: dict[str, Any],
    ) -> None:
        mock_request.return_value = response

        call_client(OsmosisClient())

        assert mock_request.call_args.kwargs["workspace_id"] == "ws_123"


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
        client.complete_upload(file_id="a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/complete"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="a/b", workspace_id="ws_test")
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
        client.get_dataset("a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_dataset_download_url_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "presigned_url": "https://example.com/download",
            "expires_in": 3600,
            "file_name": "f.jsonl",
        }
        client = OsmosisClient()
        result = client.get_dataset_download_url("a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/download"
        assert result.presigned_url == "https://example.com/download"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_delete_dataset_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.delete_dataset("a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_training_run_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run": {"id": "r1", "status": "running"},
        }
        client = OsmosisClient()
        client.get_training_run("a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb"


class TestGetModelAffectedResources:
    """Tests for OsmosisClient.get_model_affected_resources."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_model_with_blocking_runs(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_runs_using_model": [
                {"id": "r1", "training_run_name": "Run 1"},
            ],
        }
        client = OsmosisClient()
        result = client.get_model_affected_resources("m1", workspace_id="ws_test")
        assert len(result.training_runs_using_model) == 1
        assert result.has_blocking_runs is True
        path = mock_request.call_args[0][0]
        assert "/api/cli/models/m1/affected-resources" in path

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_path_encodes_model_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_runs_using_model": [],
        }
        client = OsmosisClient()
        client.get_model_affected_resources("a/b", workspace_id="ws_test")
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


class TestRefreshWorkspaceInfo:
    """Tests for OsmosisClient.refresh_workspace_info."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_returns_git_metadata_for_matching_workspace(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "workspaces": [
                {
                    "id": "ws-1",
                    "name": "team-alpha",
                    "has_subscription": True,
                    "has_github_app_installation": True,
                    "connected_repo": {
                        "id": "repo-1",
                        "repo_full_name": "acme/rollouts",
                        "repo_url": "https://github.com/acme/rollouts",
                        "default_branch": "main",
                        "sync_status": "ready",
                        "last_synced_commit_sha": "abcdef123456",
                    },
                }
            ]
        }
        client = OsmosisClient()
        result = client.refresh_workspace_info(workspace_name="team-alpha")
        assert result["found"] is True
        assert result["has_subscription"] is True
        assert result["has_github_app_installation"] is True
        assert (
            result["connected_repo"]["repo_url"] == "https://github.com/acme/rollouts"
        )
        call_kwargs = mock_request.call_args
        assert call_kwargs.args[0] == "/api/cli/workspaces"
        assert call_kwargs.kwargs["require_workspace"] is False

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_defaults_new_fields_for_older_platform_response(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "workspaces": [
                {
                    "id": "ws-1",
                    "name": "team-alpha",
                    "has_subscription": True,
                }
            ]
        }
        client = OsmosisClient()
        result = client.refresh_workspace_info(workspace_name="team-alpha")
        assert result == {
            "found": True,
            "id": "ws-1",
            "name": "team-alpha",
            "has_subscription": True,
            "has_github_app_installation": False,
            "connected_repo": None,
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_returns_found_false_when_workspace_missing(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {"workspaces": []}
        client = OsmosisClient()
        result = client.refresh_workspace_info(workspace_name="missing")
        assert result == {"found": False}

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_workspace_id_takes_precedence_over_workspace_name(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "workspaces": [
                {
                    "id": "ws-1",
                    "name": "team-alpha",
                    "has_subscription": True,
                },
                {
                    "id": "ws-2",
                    "name": "team-beta",
                    "has_subscription": False,
                },
            ]
        }
        client = OsmosisClient()

        result = client.refresh_workspace_info(
            workspace_id="ws-2", workspace_name="team-alpha"
        )

        assert result["found"] is True
        assert result["id"] == "ws-2"
        assert result["name"] == "team-beta"


class TestSubmitTrainingRun:
    """Tests for OsmosisClient.submit_training_run payload assembly."""

    @staticmethod
    def _response() -> dict[str, Any]:
        return {
            "id": "run-1",
            "name": "run-1",
            "status": "pending",
            "created_at": "2026-05-04T00:00:00Z",
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_minimal_payload_omits_optional_fields(
        self, mock_request: MagicMock
    ) -> None:
        """Optional fields are omitted from the payload when not provided."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        result = client.submit_training_run(
            model_path="m1",
            dataset="ds1",
            rollout_name="rollout1",
            entrypoint="rollouts/main.py",
            workspace_id="ws_test",
        )
        assert result.id == "run-1"
        payload = mock_request.call_args.kwargs["data"]
        assert payload == {
            "model_path": "m1",
            "dataset": "ds1",
            "rollout_name": "rollout1",
            "entrypoint": "rollouts/main.py",
        }
        assert "rollout_env" not in payload
        assert "rollout_secret_refs" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_rollout_env_included_when_non_empty(self, mock_request: MagicMock) -> None:
        """Non-empty rollout_env map is forwarded to the platform."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        rollout_env = {"FOO": "bar", "BAZ": "qux"}
        client.submit_training_run(
            model_path="m1",
            dataset="ds1",
            rollout_name="rollout1",
            entrypoint="rollouts/main.py",
            rollout_env=rollout_env,
            workspace_id="ws_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["rollout_env"] == rollout_env
        assert "rollout_secret_refs" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_rollout_secret_refs_included_when_non_empty(
        self, mock_request: MagicMock
    ) -> None:
        """Non-empty rollout_secret_refs map is forwarded to the platform."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        secret_refs = {"OPENAI_API_KEY": "openai-prod"}
        client.submit_training_run(
            model_path="m1",
            dataset="ds1",
            rollout_name="rollout1",
            entrypoint="rollouts/main.py",
            rollout_secret_refs=secret_refs,
            workspace_id="ws_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["rollout_secret_refs"] == secret_refs
        assert "rollout_env" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_empty_rollout_env_and_secret_refs_are_omitted(
        self, mock_request: MagicMock
    ) -> None:
        """Empty dicts are treated as 'not provided' and stripped from payload."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        client.submit_training_run(
            model_path="m1",
            dataset="ds1",
            rollout_name="rollout1",
            entrypoint="rollouts/main.py",
            rollout_env={},
            rollout_secret_refs={},
            workspace_id="ws_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert "rollout_env" not in payload
        assert "rollout_secret_refs" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_commit_sha_and_config_included_when_provided(
        self, mock_request: MagicMock
    ) -> None:
        """commit_sha and config are forwarded when provided."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        client.submit_training_run(
            model_path="m1",
            dataset="ds1",
            rollout_name="rollout1",
            entrypoint="rollouts/main.py",
            commit_sha="abc123",
            config={"lr": 0.001},
            rollout_env={"FOO": "bar"},
            rollout_secret_refs={"OPENAI_API_KEY": "openai-prod"},
            workspace_id="ws_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["commit_sha"] == "abc123"
        assert payload["config"] == {"lr": 0.001}
        assert payload["rollout_env"] == {"FOO": "bar"}
        assert payload["rollout_secret_refs"] == {"OPENAI_API_KEY": "openai-prod"}


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
        result = client.get_training_run_metrics("run-1", workspace_id="ws_test")
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
        client.get_training_run_metrics("a/b", workspace_id="ws_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb/metrics"
