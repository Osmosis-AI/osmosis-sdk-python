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
        },
        credentials=None,
        git_identity="acme/repo",
    )


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
