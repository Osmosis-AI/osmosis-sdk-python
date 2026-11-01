"""Platform upload orchestration for completed local evaluations."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osmosis_ai.eval.local.upload import (
    EvalUploadFile,
    EvalUploadPlan,
    LocalEvalUploadError,
    _hash_file,
)
from osmosis_ai.eval.local.upload_archive import (
    ARCHIVE_PATH,
    FILE_LIST_PATH,
    build_run_archive,
)
from osmosis_ai.platform.api.models import (
    EvalRunImportExtraction,
    EvalRunImportResult,
    EvalRunImportUploads,
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


def test_existing_single_segment_run_directory_remains_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    direct = workspace / "run-1"
    direct.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    assert _resolve_run_dir(Path("run-1"), workspace_directory=workspace) == direct


def test_existing_single_segment_directory_is_relative_to_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "tools"
    direct = nested / "run-1"
    direct.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert _resolve_run_dir(Path("run-1"), workspace_directory=workspace) == direct


def test_existing_direct_directory_wins_over_default_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    direct = workspace / "run-1"
    default = workspace / ".osmosis" / "evals" / "run-1"
    direct.mkdir(parents=True)
    default.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    assert _resolve_run_dir(Path("run-1"), workspace_directory=workspace) == direct


def _plan(tmp_path: Path, bodies: dict[str, bytes]) -> EvalUploadPlan:
    run_dir = tmp_path / "run-1"
    files = []
    for name in sorted(bodies):
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bodies[name])
        files.append(_hash_file(path, relative=name))
    return EvalUploadPlan(
        run_dir=run_dir,
        local_run_id="a" * 32,
        manifest_digest="b" * 64,
        run={"name": "run-1"},
        schema_versions={"state_schema": 1},
        provenance={},
        files=tuple(files),
    )


BODIES = {
    "index.jsonl": b'{"rollout_id": "r"}\n' * 20,
    "progress.json": b'{"total_runs": 1}\n',
    "rollout_trials/r/artifacts/a.bin": bytes(range(256)) * 40,
    "rollout_trials/r/artifacts/b.bin": bytes(reversed(range(256))) * 40,
    "rollout_trials/r/artifacts/c.bin": b"c" * 10_000,
}
CONTEXT = SimpleNamespace(credentials=None, git_identity="acme/repo")


class FakeImportServer:
    """Archive-mode import routes with the platform's slot bookkeeping."""

    def __init__(
        self, *, part_size: int = 4096, extraction: list[str] | None = None
    ) -> None:
        self.part_size = part_size
        self.extraction = list(extraction or ["pending", "ready"])
        self.slots: dict[str, dict[str, Any]] = {}
        self.archive: dict[str, Any] = {}
        self.status = "uploading"
        self.upload_mode = "archive"
        self.calls: list[tuple[Any, ...]] = []

    def _session(self, **extra: Any) -> EvalRunImportResult:
        finalized = self.status == "finalized"
        return EvalRunImportResult.from_dict(
            {
                "session_id": "session-1",
                "status": self.status,
                "upload_mode": self.upload_mode,
                "expected_files": self.archive["file_count"],
                "uploaded_files": self.archive["file_count"] if finalized else 0,
                "files": [{"path": path, **slot} for path, slot in self.slots.items()],
                **extra,
            }
        )

    def start_eval_run_import(self, **kwargs: Any) -> EvalRunImportResult:
        self.calls.append(("start", kwargs["archive"], kwargs["reimport"]))
        if not self.slots:
            self.archive = kwargs["archive"]
            file_list = self.archive["file_list"]
            self.slots = {
                FILE_LIST_PATH: {
                    "size": file_list["size"],
                    "sha256": file_list["sha256"],
                    "state": "pending",
                },
                ARCHIVE_PATH: {
                    "size": self.archive["size"],
                    "sha256": None,
                    "state": "pending",
                },
            }
        return self._session()

    def get_eval_run_import_uploads(
        self, _session_id: str, *, paths: list[str], **_kwargs: Any
    ) -> EvalRunImportUploads:
        self.calls.append(("uploads", list(paths)))
        files = []
        for path in paths:
            slot = self.slots[path]
            if path == ARCHIVE_PATH:
                total_parts = -(-slot["size"] // self.part_size)
                upload = {
                    "method": "multipart",
                    "part_size": self.part_size,
                    "total_parts": total_parts,
                    "presigned_urls": [
                        {
                            "part_number": number,
                            "presigned_url": f"https://uploads.example/run.zip/{number}",
                            "size": min(
                                self.part_size,
                                slot["size"] - (number - 1) * self.part_size,
                            ),
                        }
                        for number in range(1, total_parts + 1)
                    ],
                }
            else:
                upload = {
                    "method": "simple",
                    "presigned_url": "https://uploads.example/files.jsonl",
                }
            files.append(
                {
                    "path": path,
                    "size": slot["size"],
                    "sha256": slot["sha256"],
                    "upload": upload,
                }
            )
        return EvalRunImportUploads.from_dict({"files": files})

    def complete_eval_run_import_upload(
        self,
        _session_id: str,
        *,
        path: str,
        parts: list[dict[str, Any]] | None,
        **_kwargs: Any,
    ) -> None:
        self.calls.append(("complete", path, parts))
        self.slots[path]["state"] = "uploaded"

    def _extraction(self) -> EvalRunImportExtraction:
        status = self.extraction.pop(0)
        if status == "ready":
            return EvalRunImportExtraction.from_dict(
                {
                    "status": "ready",
                    "file_count": self.archive["file_count"],
                    "total_bytes": self.archive["total_bytes"],
                }
            )
        if status == "failed":
            return EvalRunImportExtraction.from_dict(
                {"status": "failed", "reason": "index.jsonl:3 is not valid JSON"}
            )
        return EvalRunImportExtraction.from_dict({"status": status})

    def start_eval_run_import_extraction(
        self, _session_id: str, **_kwargs: Any
    ) -> EvalRunImportExtraction:
        self.calls.append(("extract",))
        return self._extraction()

    def get_eval_run_import_extraction(
        self, _session_id: str, **_kwargs: Any
    ) -> EvalRunImportExtraction:
        self.calls.append(("poll",))
        return self._extraction()

    def finalize_eval_run_import(
        self, _session_id: str, **_kwargs: Any
    ) -> EvalRunImportResult:
        self.calls.append(("finalize",))
        self.status = "finalized"
        return self._session(
            eval_run_id="eval-1",
            eval_run_name="run-1",
            platform_url="https://platform.example/evals/eval-1",
        )


class FakeStorage:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.bodies: dict[str, bytes] = {}
        self.fail_url: str | None = None

    def put(self, url: str, *, data: Any, **_kwargs: Any) -> SimpleNamespace:
        if url == self.fail_url:
            raise RuntimeError("Upload failed: HTTP 403")
        self.events.append(("put", url))
        self.bodies[url] = data.read()
        return SimpleNamespace(headers={"etag": f'"etag-{len(self.bodies)}"'})

    def archive_parts(self) -> list[bytes]:
        urls = sorted(
            (url for url in self.bodies if "/run.zip/" in url),
            key=lambda url: int(url.rsplit("/", 1)[1]),
        )
        return [self.bodies[url] for url in urls]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("osmosis_ai.platform.cli.eval_upload.time", fake)
    return fake


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    log: list[tuple[str, str]] = []
    open_verified = EvalUploadFile.open_verified

    def record_open(file: EvalUploadFile) -> Any:
        log.append(("open", file.path))
        return open_verified(file)

    monkeypatch.setattr(EvalUploadFile, "open_verified", record_open)
    return log


@pytest.fixture
def storage(
    monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, str]]
) -> FakeStorage:
    fake = FakeStorage(events)
    monkeypatch.setattr(
        "osmosis_ai.platform.api.upload._http_put_with_backoff", fake.put
    )
    return fake


def test_upload_streams_the_archive_one_part_at_a_time(
    tmp_path: Path, storage: FakeStorage, events: list[tuple[str, str]]
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer(part_size=4096)

    result = upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    file_list = storage.bodies["https://uploads.example/files.jsonl"]
    assert [json.loads(line) for line in file_list.decode().splitlines()] == [
        {"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        for name, body in sorted(BODIES.items())
    ]
    parts = storage.archive_parts()
    archive_bytes = b"".join(parts)
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    assert archive.testzip() is None
    assert {name: archive.read(name) for name in archive.namelist()} == BODIES
    assert server.calls[0] == (
        "start",
        {
            "size": len(archive_bytes),
            "file_count": len(BODIES),
            "total_bytes": sum(len(body) for body in BODIES.values()),
            "file_list": {
                "size": len(file_list),
                "sha256": hashlib.sha256(file_list).hexdigest(),
            },
        },
        False,
    )
    # One part is buffered at a time: the first part is sent before the last
    # file is read, and no part exceeds the server's part size.
    assert len(parts) == -(-len(archive_bytes) // 4096) > 2
    assert max(len(part) for part in parts) == 4096
    first_put = events.index(("put", "https://uploads.example/run.zip/1"))
    last_open = events.index(("open", "rollout_trials/r/artifacts/c.bin"))
    assert first_put < last_open
    assert [call[0] for call in server.calls] == [
        "start",
        "uploads",
        "complete",
        "complete",
        "extract",
        "poll",
        "finalize",
    ]
    assert server.calls[1] == ("uploads", [FILE_LIST_PATH, ARCHIVE_PATH])
    assert server.calls[2] == ("complete", FILE_LIST_PATH, None)
    assert server.calls[3] == (
        "complete",
        ARCHIVE_PATH,
        [
            {"PartNumber": number, "ETag": f"etag-{number + 1}"}
            for number in range(1, len(parts) + 1)
        ],
    )
    assert result.status == "finalized"
    assert result.upload_mode == "archive"
    assert result.expected_files == result.uploaded_files == len(BODIES)


def test_file_changed_after_planning_aborts_without_completing_the_archive(
    tmp_path: Path, storage: FakeStorage
) -> None:
    plan = _plan(tmp_path, BODIES)
    (plan.run_dir / "rollout_trials/r/artifacts/c.bin").write_bytes(b"x" * 10_000)
    server = FakeImportServer(part_size=4096)

    with pytest.raises(LocalEvalUploadError, match=r"c\.bin changed after"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    assert storage.archive_parts()
    assert ("complete", ARCHIVE_PATH) not in [call[:2] for call in server.calls]
    assert server.calls[-1][:2] == ("complete", FILE_LIST_PATH)


def test_resume_uploads_only_the_slots_still_pending(
    tmp_path: Path, storage: FakeStorage
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer(part_size=4096)
    storage.fail_url = "https://uploads.example/run.zip/2"
    with pytest.raises(RuntimeError, match="HTTP 403"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]
    assert server.slots[FILE_LIST_PATH]["state"] == "uploaded"
    storage.fail_url = None
    server.calls.clear()

    result = upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    assert result.status == "finalized"
    assert server.calls[1] == ("uploads", [ARCHIVE_PATH])
    assert [call[:2] for call in server.calls if call[0] == "complete"] == [
        ("complete", ARCHIVE_PATH)
    ]


def test_resume_restarts_a_failed_extraction_without_reuploading(
    tmp_path: Path, storage: FakeStorage
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer(extraction=["pending", "failed"])
    with pytest.raises(RuntimeError, match=r"index\.jsonl:3 is not valid JSON"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]
    assert ("finalize",) not in server.calls
    server.calls.clear()
    server.extraction = ["pending", "ready"]
    uploaded = dict(storage.bodies)

    result = upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    assert result.status == "finalized"
    assert [call[0] for call in server.calls] == [
        "start",
        "extract",
        "poll",
        "finalize",
    ]
    assert storage.bodies == uploaded


def test_extraction_poll_times_out_with_a_clear_message(
    tmp_path: Path, clock: FakeClock
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer(extraction=["pending"] * 1_000)
    server.start_eval_run_import(
        archive=build_run_archive(plan.files).to_request(), reimport=False
    )
    for slot in server.slots.values():
        slot["state"] = "uploaded"

    with pytest.raises(RuntimeError, match="within 20 minutes; re-run the upload"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    assert set(clock.sleeps) == {5.0}
    assert clock.now == 20 * 60
    assert ("finalize",) not in server.calls


def test_failed_extraction_reason_reaches_the_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: FakeStorage
) -> None:
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.cli import eval_upload as eval_upload_module

    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer(extraction=["failed"])
    monkeypatch.setattr(eval_upload_module, "RunLock", lambda _path: nullcontext())
    monkeypatch.setattr(
        eval_upload_module,
        "require_git_workspace_directory_context",
        lambda: SimpleNamespace(workspace_directory=tmp_path, **vars(CONTEXT)),
    )
    monkeypatch.setattr(
        eval_upload_module, "prepare_eval_upload_plan", lambda _run_dir: plan
    )
    monkeypatch.setattr(eval_upload_module, "OsmosisClient", lambda: server)

    with pytest.raises(CLIError) as excinfo:
        eval_upload_module.upload(plan.run_dir)

    assert str(excinfo.value) == (
        "Local evaluation upload failed: The platform could not extract the "
        "uploaded files: index.jsonl:3 is not valid JSON. Re-run the upload to "
        "try again"
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda server: setattr(server, "upload_mode", "files"),
        lambda server: server.slots[ARCHIVE_PATH].update(sha256="c" * 64),
        lambda server: server.slots[FILE_LIST_PATH].update(sha256="c" * 64),
        lambda server: server.slots[ARCHIVE_PATH].update(size=1),
        lambda server: server.slots.pop(FILE_LIST_PATH),
        lambda server: server.archive.update(file_count=1),
    ],
    ids=[
        "files-mode",
        "archive-sha256",
        "file-list-sha256",
        "archive-size",
        "missing-slot",
        "file-count",
    ],
)
def test_start_response_must_match_the_archive_slots(
    tmp_path: Path, corrupt: Any
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer()
    server.start_eval_run_import(
        archive=build_run_archive(plan.files).to_request(), reimport=False
    )
    corrupt(server)
    server.calls.clear()

    with pytest.raises(RuntimeError, match="Server returned"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]
    assert [call[0] for call in server.calls] == ["start"]


def test_unknown_upload_instruction_fails_closed(
    tmp_path: Path, storage: FakeStorage
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer()
    server.start_eval_run_import(
        archive=build_run_archive(plan.files).to_request(), reimport=False
    )
    server.slots[FILE_LIST_PATH]["state"] = "uploaded"
    server.calls.clear()
    real_uploads = server.get_eval_run_import_uploads

    def uploads(session_id: str, *, paths: list[str], **kwargs: Any) -> Any:
        return real_uploads(session_id, paths=[FILE_LIST_PATH, *paths], **kwargs)

    server.get_eval_run_import_uploads = uploads  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"unknown path 'files\.jsonl'"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]
    assert storage.bodies == {}


def test_finalized_start_is_confirmed_by_idempotent_finalize(tmp_path: Path) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer()
    server.start_eval_run_import(
        archive=build_run_archive(plan.files).to_request(), reimport=False
    )
    for slot in server.slots.values():
        slot["state"] = "uploaded"
    server.status = "finalized"
    server.calls.clear()

    result = upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]

    assert result.eval_run_id == "eval-1"
    assert [call[0] for call in server.calls] == ["start", "finalize"]


def test_incomplete_finalized_response_is_rejected(
    tmp_path: Path, storage: FakeStorage
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer()
    finalize = server.finalize_eval_run_import

    def incomplete(session_id: str, **kwargs: Any) -> EvalRunImportResult:
        return dataclasses.replace(finalize(session_id, **kwargs), uploaded_files=1)

    server.finalize_eval_run_import = incomplete  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="incomplete files"):
        upload_plan(plan, context=CONTEXT, client=server)  # type: ignore[arg-type]


@pytest.mark.parametrize("replace", [False, True])
def test_replace_is_forwarded_to_the_import_start(
    tmp_path: Path, storage: FakeStorage, replace: bool
) -> None:
    plan = _plan(tmp_path, BODIES)
    server = FakeImportServer()

    upload_plan(
        plan,
        context=CONTEXT,  # type: ignore[arg-type]
        client=server,  # type: ignore[arg-type]
        replace=replace,
    )

    assert server.calls[0][2] is replace


def test_conflict_without_replace_keeps_platform_details_and_quotes_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.cli.errors import CLIError
    from osmosis_ai.platform.auth.platform_client import PlatformAPIError
    from osmosis_ai.platform.cli.eval_upload import upload as upload_run

    class DummyLock:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> DummyLock:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.eval_upload.RunLock",
        DummyLock,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.eval_upload.require_git_workspace_directory_context",
        lambda: SimpleNamespace(
            workspace_directory=tmp_path,
            credentials=None,
            git_identity="acme/repo",
        ),
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.eval_upload.prepare_eval_upload_plan",
        lambda _run_dir: SimpleNamespace(),
    )

    def _upload_plan(*_args: Any, **_kwargs: Any) -> None:
        raise PlatformAPIError(
            "already imported",
            status_code=409,
            error_code="EVAL_IMPORT_EXISTS",
            details={"local_run_id": "abc"},
        )

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.eval_upload.upload_plan",
        _upload_plan,
    )

    with pytest.raises(CLIError) as excinfo:
        upload_run(Path("my run"), replace=False)

    err = excinfo.value
    assert err.code == "CONFLICT"
    assert err.details["platform_code"] == "EVAL_IMPORT_EXISTS"
    assert err.details["local_run_id"] == "abc"
    assert "osmosis eval upload 'my run' --replace" in str(err)


@pytest.mark.parametrize("output_flag", ["--json", "--plain"])
def test_standalone_upload_errors_never_print_an_ambient_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output_flag: str,
) -> None:
    import json

    import osmosis_ai.cli.main as cli
    import osmosis_ai.platform.cli.eval_upload as eval_upload_module

    secret = "synth-upload-secret-12345"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "logs.txt").write_text(f"before {secret} after\n", encoding="utf-8")
    # A run name copied from the key: the mismatch error quotes it.
    manifest = json.dumps(
        {"schema_version": 1, "local_run_id": "a" * 32, "run_name": secret}
    ).encode()
    (run_dir / "manifest.json").write_bytes(manifest)
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        eval_upload_module,
        "require_git_workspace_directory_context",
        lambda: SimpleNamespace(
            workspace_directory=workspace,
            git_identity="example/fixture",
            repo_url="https://github.com/example/fixture.git",
            credentials=SimpleNamespace(access_token="synthetic-auth"),
        ),
    )

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("validation must fail before any upload")

    monkeypatch.setattr(eval_upload_module, "upload_plan", refuse)
    monkeypatch.chdir(workspace)

    exit_code = cli.main(
        ["--env-file", str(env_file), output_flag, "eval", "upload", str(run_dir)]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    # The command really read the key: its log pass removed it.
    assert (run_dir / "logs.txt").read_text() == "before [REDACTED] after\n"
    assert (run_dir / "manifest.json").read_bytes() == manifest
    assert secret not in captured.out + captured.err
    assert "does not match directory name 'run-1'" in captured.err
    if output_flag == "--json":
        assert json.loads(captured.err.splitlines()[-1])["error"]["code"] == (
            "VALIDATION"
        )
