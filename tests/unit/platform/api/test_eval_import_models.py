"""Strict response parsing for local-eval import sessions."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.api.models import (
    EvalRunImportResult,
    EvalRunImportUploads,
)

FILE = {"path": "index.jsonl", "size": 10, "sha256": "a" * 64}


def test_import_result_parses_strict_file_state() -> None:
    result = EvalRunImportResult.from_dict(
        {
            "session_id": "session-1",
            "status": "uploading",
            "expected_files": 1,
            "uploaded_files": 0,
            "files": [{**FILE, "state": "pending"}],
            "expires_at": None,
        }
    )
    assert result.files[0].path == "index.jsonl"


@pytest.mark.parametrize(
    "change",
    [
        {"files": ["not-an-object"]},
        {"files": [{**FILE, "sha256": "bad", "state": "pending"}]},
        {"files": [{**FILE, "state": "unknown"}]},
        {"files": [{**FILE, "state": "pending"}, {**FILE, "state": "uploaded"}]},
        {"platform_url": 123},
    ],
)
def test_import_result_rejects_malformed_fields(change: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "session_id": "session-1",
        "status": "uploading",
        "expected_files": 1,
        "uploaded_files": 0,
        "files": [{**FILE, "state": "pending"}],
    }
    payload.update(change)
    with pytest.raises(ValueError):
        EvalRunImportResult.from_dict(payload)  # type: ignore[arg-type]


def test_upload_instructions_reject_duplicate_paths() -> None:
    upload = {"method": "simple", "presigned_url": "https://example.test/upload"}
    with pytest.raises(ValueError, match="Duplicate"):
        EvalRunImportUploads.from_dict(
            {
                "files": [
                    {**FILE, "upload": upload},
                    {**FILE, "upload": upload},
                ]
            }
        )
