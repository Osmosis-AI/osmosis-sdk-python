"""Strict response parsing for local-eval import sessions."""

from __future__ import annotations

import pytest

from osmosis_ai.platform.api.models import (
    EvalRunImportExtraction,
    EvalRunImportResult,
    EvalRunImportUploads,
)

FILE = {"path": "files.jsonl", "size": 10, "sha256": "a" * 64}
ARCHIVE = {"path": "run.zip", "size": 4096, "sha256": None}


def test_archive_import_result_parses_slots_and_run_file_counts() -> None:
    result = EvalRunImportResult.from_dict(
        {
            "session_id": "session-1",
            "status": "uploading",
            "upload_mode": "archive",
            "expected_files": 9_338,
            "uploaded_files": 0,
            "files": [{**FILE, "state": "uploaded"}, {**ARCHIVE, "state": "pending"}],
            "expires_at": None,
        }
    )
    assert result.upload_mode == "archive"
    assert [(file.path, file.sha256) for file in result.files] == [
        ("files.jsonl", "a" * 64),
        ("run.zip", None),
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"files": ["not-an-object"]},
        {"files": [{**FILE, "sha256": "bad", "state": "pending"}]},
        {"files": [{**FILE, "state": "unknown"}]},
        {"files": [{**FILE, "state": "pending"}, {**FILE, "state": "uploaded"}]},
        {"platform_url": 123},
        {"upload_mode": None},
        {"upload_mode": "zip"},
    ],
)
def test_import_result_rejects_malformed_fields(change: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "session_id": "session-1",
        "status": "uploading",
        "upload_mode": "archive",
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


def test_upload_instruction_accepts_a_null_archive_sha256() -> None:
    upload = {"method": "multipart", "part_size": 4096, "total_parts": 1}
    parsed = EvalRunImportUploads.from_dict({"files": [{**ARCHIVE, "upload": upload}]})
    assert parsed.files[0].sha256 is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "pending"}, EvalRunImportExtraction(status="pending")),
        (
            {"status": "ready", "file_count": 9_338, "total_bytes": 612_345_678},
            EvalRunImportExtraction(
                status="ready", file_count=9_338, total_bytes=612_345_678
            ),
        ),
        (
            {"status": "failed", "reason": "run.zip is not a valid archive"},
            EvalRunImportExtraction(
                status="failed", reason="run.zip is not a valid archive"
            ),
        ),
    ],
)
def test_extraction_state_parses_each_status(
    payload: dict[str, object], expected: EvalRunImportExtraction
) -> None:
    assert EvalRunImportExtraction.from_dict(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "running"},
        {"status": "ready", "file_count": 1},
        {"status": "ready", "file_count": -1, "total_bytes": 1},
        {"status": "failed"},
        {"status": "failed", "reason": ""},
    ],
)
def test_extraction_state_rejects_malformed_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        EvalRunImportExtraction.from_dict(payload)
