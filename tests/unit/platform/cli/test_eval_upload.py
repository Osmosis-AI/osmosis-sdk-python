"""Platform upload orchestration for completed local evaluations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osmosis_ai.eval.local.upload import EvalUploadFile, EvalUploadPlan
from osmosis_ai.platform.api.models import (
    EvalRunImportFileStatus,
    EvalRunImportResult,
    EvalRunImportUploadInstruction,
    EvalRunImportUploads,
    UploadInfo,
)
from osmosis_ai.platform.cli.eval_upload import _resolve_run_dir, upload_plan


def test_run_name_resolves_under_the_default_workspace_eval_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    assert (
        _resolve_run_dir(Path("patient-finch-73"), workspace_directory=workspace)
        == workspace / ".osmosis" / "evals" / "patient-finch-73"
    )


def test_explicit_run_directory_remains_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    custom = workspace / "custom-evals" / "run-1"
    custom.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    assert (
        _resolve_run_dir(Path("custom-evals/run-1"), workspace_directory=workspace)
        == custom
    )


def test_explicit_run_directory_is_relative_to_the_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "tools"
    custom = nested / "custom-evals" / "run-1"
    custom.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert (
        _resolve_run_dir(Path("custom-evals/run-1"), workspace_directory=workspace)
        == custom
    )


def test_single_segment_is_unambiguously_a_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    direct = workspace / "run-1"
    direct.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    assert (
        _resolve_run_dir(Path("run-1"), workspace_directory=workspace)
        == workspace / ".osmosis" / "evals" / "run-1"
    )


def _local_file(tmp_path: Path, name: str, body: bytes) -> EvalUploadFile:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    identity = path.stat()
    return EvalUploadFile(
        path=name,
        source=path,
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        device=identity.st_dev,
        inode=identity.st_ino,
        modified_ns=identity.st_mtime_ns,
        changed_ns=identity.st_ctime_ns,
    )


def _result(
    files: tuple[EvalUploadFile, ...], *, finalized: bool
) -> EvalRunImportResult:
    return EvalRunImportResult(
        session_id="session-1",
        status="finalized" if finalized else "uploading",
        expected_files=len(files),
        uploaded_files=len(files) if finalized else 1,
        files=[
            EvalRunImportFileStatus(
                path=file.path,
                size=file.size,
                sha256=file.sha256,
                state=(
                    "uploaded" if finalized or file.path == "index.jsonl" else "pending"
                ),
            )
            for file in files
        ],
        eval_run_id="eval-1" if finalized else None,
        eval_run_name="run-1" if finalized else None,
        platform_url="https://platform.example/evals/eval-1" if finalized else None,
    )


def test_server_uploaded_files_are_skipped_and_missing_files_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = (
        _local_file(tmp_path, "index.jsonl", b"{}\n"),
        _local_file(tmp_path, "progress.json", b'{"total_runs":1}\n'),
    )
    plan = EvalUploadPlan(
        run_dir=tmp_path,
        local_run_id="a" * 32,
        manifest_digest="b" * 64,
        run={"name": "run-1"},
        schema_versions={"state_schema": 1},
        provenance={},
        files=files,
    )

    class FakeClient:
        completed: list[tuple[str, list[dict[str, Any]] | None]] = []

        def start_eval_run_import(self, **_kwargs: Any) -> EvalRunImportResult:
            return _result(files, finalized=False)

        def get_eval_run_import_uploads(
            self, _session_id: str, *, paths: list[str], **_kwargs: Any
        ) -> EvalRunImportUploads:
            assert paths == ["index.jsonl", "progress.json"]
            file = files[1]
            return EvalRunImportUploads(
                files=[
                    EvalRunImportUploadInstruction(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        upload=UploadInfo(
                            method="simple",
                            presigned_url="https://uploads.example/progress",
                        ),
                    )
                ]
            )

        def complete_eval_run_import_upload(
            self,
            _session_id: str,
            *,
            path: str,
            parts: list[dict[str, Any]] | None,
            **_kwargs: Any,
        ) -> None:
            self.completed.append((path, parts))

        def finalize_eval_run_import(
            self, _session_id: str, **_kwargs: Any
        ) -> EvalRunImportResult:
            return _result(files, finalized=True)

    uploaded: list[bytes] = []
    monkeypatch.setattr(
        "osmosis_ai.platform.api.upload._upload_fileobj_simple",
        lambda handle, _size, _upload: uploaded.append(handle.read()),
    )
    context = SimpleNamespace(
        credentials=object(),
        git_identity="acme/repo",
        repo_url="https://github.com/acme/repo",
        workspace_directory=tmp_path,
    )

    result = upload_plan(plan, context=context, client=FakeClient())  # type: ignore[arg-type]

    assert result.status == "finalized"
    assert uploaded == [b'{"total_runs":1}\n']
    assert FakeClient.completed == [("progress.json", None)]


def test_unknown_upload_instruction_fails_closed(tmp_path: Path) -> None:
    file = _local_file(tmp_path, "index.jsonl", b"{}\n")
    plan = EvalUploadPlan(
        run_dir=tmp_path,
        local_run_id="a" * 32,
        manifest_digest="b" * 64,
        run={"name": "run-1"},
        schema_versions={},
        provenance={},
        files=(file,),
    )

    class FakeClient:
        def start_eval_run_import(self, **_kwargs: Any) -> EvalRunImportResult:
            return EvalRunImportResult(
                session_id="session-1",
                status="uploading",
                expected_files=1,
                uploaded_files=0,
                files=[
                    EvalRunImportFileStatus(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        state="pending",
                    )
                ],
            )

        def get_eval_run_import_uploads(
            self, *_args: Any, **_kwargs: Any
        ) -> EvalRunImportUploads:
            return EvalRunImportUploads(
                files=[
                    EvalRunImportUploadInstruction(
                        path="unknown.txt",
                        size=1,
                        sha256="c" * 64,
                        upload=UploadInfo(
                            method="simple",
                            presigned_url="https://uploads.example/unknown",
                        ),
                    )
                ]
            )

    context = SimpleNamespace(credentials=None, git_identity="acme/repo")
    with pytest.raises(RuntimeError, match="unknown path"):
        upload_plan(plan, context=context, client=FakeClient())  # type: ignore[arg-type]


def test_finalized_start_is_confirmed_by_idempotent_finalize(tmp_path: Path) -> None:
    file = _local_file(tmp_path, "index.jsonl", b"{}\n")
    plan = EvalUploadPlan(
        run_dir=tmp_path,
        local_run_id="a" * 32,
        manifest_digest="b" * 64,
        run={"name": "run-1"},
        schema_versions={},
        provenance={},
        files=(file,),
    )

    class FakeClient:
        finalized_calls = 0

        def start_eval_run_import(self, **_kwargs: Any) -> EvalRunImportResult:
            return EvalRunImportResult(
                session_id="session-1",
                status="finalized",
                expected_files=1,
                uploaded_files=1,
                files=[
                    EvalRunImportFileStatus(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        state="uploaded",
                    )
                ],
            )

        def finalize_eval_run_import(
            self, _session_id: str, **_kwargs: Any
        ) -> EvalRunImportResult:
            self.finalized_calls += 1
            return EvalRunImportResult(
                session_id="session-1",
                status="finalized",
                expected_files=1,
                uploaded_files=1,
                files=[
                    EvalRunImportFileStatus(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        state="uploaded",
                    )
                ],
                eval_run_id="eval-1",
                eval_run_name="run-1",
                platform_url="https://platform.example/evals/eval-1",
            )

    client = FakeClient()
    context = SimpleNamespace(credentials=None, git_identity="acme/repo")

    result = upload_plan(plan, context=context, client=client)  # type: ignore[arg-type]

    assert result.eval_run_id == "eval-1"
    assert client.finalized_calls == 1


def test_incomplete_finalized_response_is_rejected(tmp_path: Path) -> None:
    file = _local_file(tmp_path, "index.jsonl", b"{}\n")
    plan = EvalUploadPlan(
        run_dir=tmp_path,
        local_run_id="a" * 32,
        manifest_digest="b" * 64,
        run={"name": "run-1"},
        schema_versions={},
        provenance={},
        files=(file,),
    )

    class FakeClient:
        def start_eval_run_import(self, **_kwargs: Any) -> EvalRunImportResult:
            return EvalRunImportResult(
                session_id="session-1",
                status="finalized",
                expected_files=1,
                uploaded_files=1,
                files=[
                    EvalRunImportFileStatus(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        state="uploaded",
                    )
                ],
            )

        def finalize_eval_run_import(
            self, _session_id: str, **_kwargs: Any
        ) -> EvalRunImportResult:
            return EvalRunImportResult(
                session_id="session-1",
                status="finalized",
                expected_files=1,
                uploaded_files=0,
                files=[
                    EvalRunImportFileStatus(
                        path=file.path,
                        size=file.size,
                        sha256=file.sha256,
                        state="pending",
                    )
                ],
                eval_run_id="eval-1",
                eval_run_name="run-1",
                platform_url="https://platform.example/evals/eval-1",
            )

    context = SimpleNamespace(credentials=None, git_identity="acme/repo")
    with pytest.raises(RuntimeError, match="incomplete files"):
        upload_plan(plan, context=context, client=FakeClient())  # type: ignore[arg-type]
