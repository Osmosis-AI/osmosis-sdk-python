"""Tests for osmosis_ai.platform.api.client.OsmosisClient."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from osmosis_ai.platform.api.client import OsmosisClient, _safe_path

REMOVED_CLIENT_METHODS = (
    "list_workspaces",
    "create_workspace",
    "delete_workspace",
    "get_workspace_deletion_status",
    "delete_dataset",
    "get_dataset_affected_resources",
    "delete_training_run",
    "delete_model",
    "get_model_affected_resources",
    "rename_checkpoint",
    "delete_deployment",
)


class TestRemovedClientMethods:
    """Destructive API methods must not be exposed by OsmosisClient."""

    @pytest.mark.parametrize("method_name", REMOVED_CLIENT_METHODS)
    def test_removed_method_is_not_exposed(self, method_name: str) -> None:
        assert not hasattr(OsmosisClient, method_name)


class TestCreateDataset:
    """Tests for OsmosisClient.create_dataset request payloads."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_create_dataset_omits_overwrite_dataset_id_by_default(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "id": "file-1",
            "file_name": "data",
            "file_size": 100,
            "status": "uploading",
        }

        OsmosisClient().create_dataset("data", 100, "jsonl", git_identity="git_test")

        payload = mock_request.call_args.kwargs["data"]
        assert payload == {
            "file_name": "data",
            "file_size": 100,
            "extension": "jsonl",
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_create_dataset_sends_overwrite_dataset_id_when_provided(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "id": "file-2",
            "file_name": "data",
            "file_size": 100,
            "status": "uploading",
        }

        OsmosisClient().create_dataset(
            "data",
            100,
            "jsonl",
            overwrite_dataset_id="file-1",
            git_identity="git_test",
        )

        payload = mock_request.call_args.kwargs["data"]
        assert payload["overwrite_dataset_id"] == "file-1"


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
            file_id="file-1", parts=None, git_identity="git_test"
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
            file_id="file-2", parts=parts, git_identity="git_test"
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
                file_id="file-1", parts=parts, git_identity="git_test"
            )


class TestAbortUpload:
    """Tests for OsmosisClient.abort_upload."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_sends_empty_body(self, mock_request: MagicMock) -> None:
        """Verify abort sends empty body (server reads upload_id from DB)."""
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="file-1", git_identity="git_test")
        call_kwargs = mock_request.call_args
        payload = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert payload == {}


class TestListDatasets:
    """Tests for OsmosisClient.list_datasets."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list_datasets_passes_git_identity(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"datasets": [], "total_count": 0}

        OsmosisClient().list_datasets(credentials=object(), git_identity="git_123")

        assert mock_request.call_args.kwargs["git_identity"] == "git_123"
        assert "workspace_id" not in mock_request.call_args.kwargs


class TestGitIdentityPassthrough:
    """Tests representative repo-scoped methods pass git_identity."""

    @pytest.mark.parametrize(
        ("call_client", "response"),
        [
            (
                lambda client: client.create_dataset(
                    "data.jsonl", 100, "jsonl", git_identity="git_123"
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
                    experiment_config={
                        "model_path": "openai/gpt-oss",
                        "dataset": "dataset-1",
                        "rollout": "rollout",
                        "entrypoint": "rollout.py",
                    },
                    git_identity="git_123",
                ),
                {
                    "id": "run-1",
                    "name": "Run 1",
                    "status": "pending",
                    "created_at": "2026-05-03T00:00:00Z",
                },
            ),
            (
                lambda client: client.list_training_runs(git_identity="git_123"),
                {"training_runs": [], "total_count": 0},
            ),
            (
                lambda client: client.list_rollouts(git_identity="git_123"),
                {"rollouts": [], "total_count": 0},
            ),
            (
                lambda client: client.list_base_models(git_identity="git_123"),
                {"models": [], "total_count": 0},
            ),
            (
                lambda client: client.list_deployments(git_identity="git_123"),
                {"deployments": [], "total_count": 0},
            ),
            (
                lambda client: client.deploy_checkpoint(
                    "checkpoint-1", git_identity="git_123"
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
                    "run-1", git_identity="git_123"
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
    def test_repo_scoped_methods_pass_git_identity(
        self,
        mock_request: MagicMock,
        call_client: Callable[[OsmosisClient], object],
        response: dict[str, Any],
    ) -> None:
        mock_request.return_value = response

        call_client(OsmosisClient())

        assert mock_request.call_args.kwargs["git_identity"] == "git_123"
        assert "workspace_id" not in mock_request.call_args.kwargs


class TestEnvironmentSecrets:
    """Tests for workspace environment secret request contracts."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list_environment_secrets_uses_pagination_and_git_scope(
        self, mock_request: MagicMock
    ) -> None:
        credentials = object()
        mock_request.return_value = {
            "environment_secrets": [
                {
                    "id": "sec-1",
                    "name": "OPENAI_API_KEY",
                    "created_at": "2026-05-01T00:00:00Z",
                    "updated_at": "2026-05-01T00:00:01Z",
                    "creator_name": "Ada",
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
            "platform_url": "https://platform.osmosis.ai/acme/secrets",
        }

        result = OsmosisClient().list_environment_secrets(
            limit=25,
            offset=50,
            credentials=credentials,
            git_identity="git_123",
        )

        assert mock_request.call_args[0][0] == (
            "/api/cli/environment-secrets?limit=25&offset=50"
        )
        assert mock_request.call_args.kwargs["credentials"] is credentials
        assert mock_request.call_args.kwargs["git_identity"] == "git_123"
        assert len(result.environment_secrets) == 1
        assert result.environment_secrets[0].name == "OPENAI_API_KEY"
        assert not hasattr(result.environment_secrets[0], "value")
        assert result.platform_url == "https://platform.osmosis.ai/acme/secrets"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_create_environment_secret_posts_value_once_and_returns_metadata(
        self, mock_request: MagicMock
    ) -> None:
        credentials = object()
        secret_value = "sk-never-return-this"
        mock_request.return_value = {
            "id": "sec-1",
            "name": "OPENAI_API_KEY",
            "created_at": "2026-05-01T00:00:00Z",
            "updated_at": "2026-05-01T00:00:01Z",
            "creator_name": "Ada",
            "platform_url": "https://platform.osmosis.ai/acme/secrets",
        }

        result = OsmosisClient().create_environment_secret(
            "OPENAI_API_KEY",
            secret_value,
            credentials=credentials,
            git_identity="git_123",
        )

        assert mock_request.call_args[0][0] == "/api/cli/environment-secrets"
        assert mock_request.call_args.kwargs["method"] == "POST"
        assert mock_request.call_args.kwargs["data"] == {
            "name": "OPENAI_API_KEY",
            "value": secret_value,
        }
        assert mock_request.call_args.kwargs["credentials"] is credentials
        assert mock_request.call_args.kwargs["git_identity"] == "git_123"
        assert result.name == "OPENAI_API_KEY"
        assert result.platform_url == "https://platform.osmosis.ai/acme/secrets"
        assert not hasattr(result, "value")


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
        client.complete_upload(file_id="a/b", git_identity="git_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/complete"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_abort_upload_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = None
        client = OsmosisClient()
        client.abort_upload(file_id="a/b", git_identity="git_test")
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
        client.get_dataset("a/b", git_identity="git_test")
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
        result = client.get_dataset_download_url("a/b", git_identity="git_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/datasets/a%2Fb/download"
        assert result.presigned_url == "https://example.com/download"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_training_run_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run": {"id": "r1", "status": "running"},
        }
        client = OsmosisClient()
        client.get_training_run("a/b", git_identity="git_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb"


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

    def test_legacy_flattened_kwargs_are_not_supported(self) -> None:
        """Training submissions use the platform's nested config contract only."""
        client = OsmosisClient()
        with pytest.raises(TypeError):
            client.submit_training_run(
                model="m1",
                dataset="ds1",
                rollout_name="rollout1",
                entrypoint="rollouts/main.py",
                git_identity="git_test",
            )

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_minimal_payload_omits_optional_fields(
        self, mock_request: MagicMock
    ) -> None:
        """Optional fields are omitted from the payload when not provided."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        result = client.submit_training_run(
            experiment_config={
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
            },
            git_identity="git_test",
        )
        assert result.id == "run-1"
        payload = mock_request.call_args.kwargs["data"]
        assert payload == {
            "experiment_config": {
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
            },
        }
        assert "env_config" not in payload
        assert "secret_refs_config" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_env_config_included_when_non_empty(self, mock_request: MagicMock) -> None:
        """Non-empty env_config map is forwarded to the platform."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        env_config = {"FOO": "bar", "BAZ": "qux"}
        client.submit_training_run(
            experiment_config={
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
            },
            env_config=env_config,
            git_identity="git_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["env_config"] == env_config
        assert "secret_refs_config" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_secret_refs_config_included_when_non_empty(
        self, mock_request: MagicMock
    ) -> None:
        """Non-empty secret_refs_config map is forwarded to the platform."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        secret_refs_config = {"OPENAI_API_KEY": "openai-prod"}
        client.submit_training_run(
            experiment_config={
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
            },
            secret_refs_config=secret_refs_config,
            git_identity="git_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["secret_refs_config"] == secret_refs_config
        assert "env_config" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_empty_env_and_secret_refs_configs_are_omitted(
        self, mock_request: MagicMock
    ) -> None:
        """Empty dicts are treated as 'not provided' and stripped from payload."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        client.submit_training_run(
            experiment_config={
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
            },
            env_config={},
            secret_refs_config={},
            git_identity="git_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert "env_config" not in payload
        assert "secret_refs_config" not in payload

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_commit_sha_and_config_sections_included_when_provided(
        self, mock_request: MagicMock
    ) -> None:
        """commit_sha and config sections are forwarded using the CLI API shape."""
        mock_request.return_value = self._response()
        client = OsmosisClient()
        client.submit_training_run(
            experiment_config={
                "model_path": "m1",
                "dataset": "ds1",
                "rollout": "rollout1",
                "entrypoint": "rollouts/main.py",
                "commit_sha": "abc123",
            },
            training_config={"lr": 0.001},
            sampling_config={"rollout_temperature": 0.8},
            checkpoints_config={"checkpoint_save_freq": 10},
            advanced_config={"optimizer": "adam"},
            env_config={"FOO": "bar"},
            secret_refs_config={"OPENAI_API_KEY": "openai-prod"},
            git_identity="git_test",
        )
        payload = mock_request.call_args.kwargs["data"]
        assert payload["experiment_config"] == {
            "model_path": "m1",
            "dataset": "ds1",
            "rollout": "rollout1",
            "entrypoint": "rollouts/main.py",
            "commit_sha": "abc123",
        }
        assert payload["training_config"] == {"lr": 0.001}
        assert payload["sampling_config"] == {"rollout_temperature": 0.8}
        assert payload["checkpoints_config"] == {"checkpoint_save_freq": 10}
        assert payload["advanced_config"] == {"optimizer": "adam"}
        assert payload["env_config"] == {"FOO": "bar"}
        assert payload["secret_refs_config"] == {"OPENAI_API_KEY": "openai-prod"}


class TestEvaluationRuns:
    """Tests for OsmosisClient evaluation run request contracts."""

    @staticmethod
    def _submit_response() -> dict[str, Any]:
        return {
            "id": "eval-1",
            "name": "eval-1",
            "status": "pending",
            "created_at": "2026-05-04T00:00:00Z",
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_submit_evaluation_run_minimal_payload_omits_optional_fields(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = self._submit_response()
        result = OsmosisClient().submit_evaluation_run(
            experiment_config={
                "model_path": "openai/gpt-oss",
                "dataset": "dataset-1",
                "rollout": "rollout",
                "entrypoint": "rollout.py",
            },
            git_identity="git_test",
        )

        assert result.id == "eval-1"
        assert mock_request.call_args[0][0] == "/api/cli/eval-runs"
        assert mock_request.call_args.kwargs["method"] == "POST"
        assert mock_request.call_args.kwargs["data"] == {
            "experiment_config": {
                "model_path": "openai/gpt-oss",
                "dataset": "dataset-1",
                "rollout": "rollout",
                "entrypoint": "rollout.py",
            }
        }
        assert mock_request.call_args.kwargs["git_identity"] == "git_test"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_submit_evaluation_run_includes_non_empty_config_sections(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = self._submit_response()

        OsmosisClient().submit_evaluation_run(
            experiment_config={"model_path": "openai/gpt-oss"},
            evaluation_config={"rubric": "grade correctness"},
            advanced_config={"max_concurrent_rollouts": 8},
            env_config={"FOO": "bar"},
            secret_refs_config={"OPENAI_API_KEY": "openai-prod"},
            git_identity="git_test",
        )

        payload = mock_request.call_args.kwargs["data"]
        assert payload == {
            "experiment_config": {"model_path": "openai/gpt-oss"},
            "evaluation_config": {"rubric": "grade correctness"},
            "advanced_config": {"max_concurrent_rollouts": 8},
            "env_config": {"FOO": "bar"},
            "secret_refs_config": {"OPENAI_API_KEY": "openai-prod"},
        }

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list_eval_runs_uses_pagination_query(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {
            "eval_runs": [
                {
                    "id": "eval-1",
                    "name": "math-eval",
                    "status": "pending",
                    "created_at": "2026-05-04T00:00:00Z",
                }
            ],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }

        result = OsmosisClient().list_eval_runs(
            limit=25,
            offset=50,
            git_identity="git_test",
        )

        assert len(result.eval_runs) == 1
        assert result.eval_runs[0].name == "math-eval"
        assert mock_request.call_args[0][0] == "/api/cli/eval-runs?limit=25&offset=50"
        assert mock_request.call_args.kwargs["git_identity"] == "git_test"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_eval_run_encodes_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "eval_run": {"id": "a/b", "status": "running"},
        }

        result = OsmosisClient().get_eval_run("a/b", git_identity="git_test")

        assert result.eval_run["id"] == "a/b"
        assert mock_request.call_args[0][0] == "/api/cli/eval-runs/a%2Fb"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_stop_eval_run_posts_empty_body_and_encodes_id(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = {"id": "a/b", "status": "stopping"}

        result = OsmosisClient().stop_eval_run("a/b", git_identity="git_test")

        assert result == {"id": "a/b", "status": "stopping"}
        assert mock_request.call_args[0][0] == "/api/cli/eval-runs/a%2Fb/stop"
        assert mock_request.call_args.kwargs["method"] == "POST"
        assert mock_request.call_args.kwargs["data"] == {}


class TestGetTrainingRunMetrics:
    """Tests for OsmosisClient.get_training_run_metrics."""

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_returns_parsed_metrics(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run_id": "run-1",
            "status": "finished",
            "overview": {
                "duration_ms": 3600000,
                "duration_formatted": "1h",
                "metric_summaries": [
                    {
                        "key": "rollout/raw_reward",
                        "title": "Training Reward",
                        "initial": 0.70,
                        "latest": 0.85,
                        "delta": 0.15,
                        "min": 0.65,
                        "max": 0.87,
                    },
                ],
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
        result = client.get_training_run_metrics("run-1", git_identity="git_test")
        assert result.training_run_id == "run-1"
        assert result.overview.metric_summaries[0].latest == 0.85
        assert len(result.metrics) == 1
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/run-1/metrics"

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_encodes_run_id(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "training_run_id": "a/b",
            "status": "finished",
            "overview": {
                "duration_ms": None,
                "duration_formatted": None,
                "metric_summaries": [],
                "examples_processed_count": None,
            },
            "metrics": [],
        }
        client = OsmosisClient()
        client.get_training_run_metrics("a/b", git_identity="git_test")
        path = mock_request.call_args[0][0]
        assert path == "/api/cli/training-runs/a%2Fb/metrics"
