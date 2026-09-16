"""Wire contract for local-eval import client methods."""

from __future__ import annotations

from unittest.mock import patch

from osmosis_ai.platform.api.client import OsmosisClient

IMPORT_RESPONSE = {
    "session_id": "session-1",
    "status": "uploading",
    "expected_files": 1,
    "uploaded_files": 0,
    "files": [
        {
            "path": "index.jsonl",
            "size": 10,
            "sha256": "a" * 64,
            "state": "pending",
        }
    ],
}


@patch("osmosis_ai.platform.api.client.platform_request")
def test_start_uses_the_run_metadata_wrapper(mock_request) -> None:
    mock_request.return_value = IMPORT_RESPONSE
    client = OsmosisClient()
    client.start_eval_run_import(
        local_run_id="b" * 32,
        manifest_digest="c" * 64,
        run={
            "name": "run-1",
            "started_at": "2026-08-18T01:00:00Z",
            "completed_at": "2026-08-18T01:01:00Z",
            "experiment_config": {},
            "evaluation_config": {},
        },
        schema_versions={"state_schema": 1},
        provenance={"sdk_version": "0.3.0"},
        files=[{"path": "index.jsonl", "size": 10, "sha256": "a" * 64}],
        git_identity="acme/repo",
    )

    mock_request.assert_called_once_with(
        "/api/cli/eval-runs/imports",
        method="POST",
        data={
            "schema_version": 1,
            "local_run_id": "b" * 32,
            "manifest_digest": "c" * 64,
            "run": {
                "name": "run-1",
                "started_at": "2026-08-18T01:00:00Z",
                "completed_at": "2026-08-18T01:01:00Z",
                "experiment_config": {},
                "evaluation_config": {},
            },
            "schema_versions": {"state_schema": 1},
            "provenance": {"sdk_version": "0.3.0"},
            "files": [{"path": "index.jsonl", "size": 10, "sha256": "a" * 64}],
            "reimport": False,
        },
        credentials=None,
        git_identity="acme/repo",
    )


@patch("osmosis_ai.platform.api.client.platform_request")
def test_start_forwards_the_reimport_opt_in(mock_request) -> None:
    mock_request.return_value = IMPORT_RESPONSE
    OsmosisClient().start_eval_run_import(
        local_run_id="b" * 32,
        manifest_digest="c" * 64,
        run={},
        schema_versions={},
        provenance={},
        files=[{"path": "index.jsonl", "size": 10, "sha256": "a" * 64}],
        reimport=True,
        git_identity="acme/repo",
    )

    assert mock_request.call_args.kwargs["data"]["reimport"] is True


@patch("osmosis_ai.platform.api.client.platform_request")
def test_complete_multipart_normalizes_upload_helper_parts(mock_request) -> None:
    OsmosisClient().complete_eval_run_import_upload(
        "session-1",
        path="rollout_trials/a/artifacts/output.bin",
        parts=[
            {"PartNumber": 1, "ETag": "etag-one"},
            {"PartNumber": 2, "ETag": "etag-two"},
        ],
        git_identity="acme/repo",
    )

    mock_request.assert_called_once_with(
        "/api/cli/eval-runs/imports/session-1/uploads/complete",
        method="POST",
        data={
            "path": "rollout_trials/a/artifacts/output.bin",
            "parts": [
                {"part_number": 1, "etag": "etag-one"},
                {"part_number": 2, "etag": "etag-two"},
            ],
        },
        timeout=120.0,
        credentials=None,
        git_identity="acme/repo",
    )


RETRY_RESPONSE = {
    "id": "11111111-1111-4111-8111-111111111111",
    "name": "brave-otter",
    "status": "pending",
    "workflow_id": "cloud-eval/11111111-1111-4111-8111-111111111111",
    "retryable_samples": 3,
    "platform_url": "https://platform.osmosis.ai/acme/eval/1",
}


@patch("osmosis_ai.platform.api.client.platform_request")
def test_retry_posts_to_the_run_retry_route(mock_request) -> None:
    mock_request.return_value = RETRY_RESPONSE
    result = OsmosisClient().retry_eval_run("brave-otter", git_identity="acme/repo")

    mock_request.assert_called_once_with(
        "/api/cli/eval-runs/brave-otter/retry",
        method="POST",
        data={},
        credentials=None,
        git_identity="acme/repo",
    )
    assert result.retryable_samples == 3
    assert result.platform_url == "https://platform.osmosis.ai/acme/eval/1"


@patch("osmosis_ai.platform.api.client.platform_request")
def test_retry_urlencodes_a_traversal_attempt_in_the_name(mock_request) -> None:
    mock_request.return_value = RETRY_RESPONSE
    OsmosisClient().retry_eval_run("../admin", git_identity="acme/repo")

    assert mock_request.call_args.args[0] == "/api/cli/eval-runs/..%2Fadmin/retry"
