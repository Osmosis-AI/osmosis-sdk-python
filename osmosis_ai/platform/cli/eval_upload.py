"""Handler for ``osmosis eval upload`` and ``eval run --upload``."""

from __future__ import annotations

import io
import os
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from shlex import quote
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.output.error import classify_error
from osmosis_ai.eval.local.runner import LOGS_FILENAME, scrub_logs_file
from osmosis_ai.eval.local.state import (
    LOCKS_DIRNAME,
    LocalEvalStateError,
    RunLock,
    validate_run_name,
)
from osmosis_ai.eval.local.upload import (
    EvalUploadPlan,
    LocalEvalUploadError,
    build_eval_upload_plan,
)
from osmosis_ai.eval.local.upload_archive import (
    ARCHIVE_PATH,
    FILE_LIST_PATH,
    RunArchive,
    RunFileReader,
    build_run_archive,
)
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import EvalRunImportResult, UploadInfo
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli.utils import (
    format_size,
    require_platform_workspace_context,
)
from osmosis_ai.platform.cli.workspace_directory_context import workspace_result_context
from osmosis_ai.platform.cli.workspace_directory_contract import (
    find_workspace_directory,
)

if TYPE_CHECKING:
    from osmosis_ai.platform.cli.workspace_directory_context import (
        GitWorkspaceDirectoryContext,
        PlatformWorkspaceContext,
    )

_EXTRACTION_POLL_SECONDS = 5.0
# The platform reports an extraction as failed 17 minutes after starting it.
_EXTRACTION_TIMEOUT_SECONDS = 20 * 60.0


def _expected_slots(archive: RunArchive) -> dict[str, tuple[int, str | None]]:
    return {
        FILE_LIST_PATH: (len(archive.file_list), archive.file_list_sha256),
        ARCHIVE_PATH: (archive.layout.size, None),
    }


def _check_slot(path: str, size: int, sha256: str | None, archive: RunArchive) -> None:
    expected = _expected_slots(archive).get(path)
    if expected is None:
        raise RuntimeError(f"Server returned unknown eval import slot {path!r}")
    if (size, sha256) != expected:
        raise RuntimeError(f"Server returned mismatched size or sha256 for {path!r}")


def _validate_session(
    result: EvalRunImportResult, plan: EvalUploadPlan, archive: RunArchive
) -> None:
    if result.upload_mode != "archive":
        raise RuntimeError(
            f"Server returned an eval import in {result.upload_mode!r} upload "
            "mode; expected 'archive'"
        )
    slots = {file.path for file in result.files}
    if slots != set(_expected_slots(archive)):
        raise RuntimeError(
            f"Server returned unexpected eval import slots {sorted(slots)[:5]}"
        )
    for slot in result.files:
        _check_slot(slot.path, slot.size, slot.sha256, archive)
    if result.expected_files != len(plan.files) or (
        result.status == "uploading" and result.uploaded_files != 0
    ):
        raise RuntimeError("Server returned inconsistent eval import file counts")


def _validate_finalized(
    result: EvalRunImportResult, plan: EvalUploadPlan, archive: RunArchive
) -> None:
    _validate_session(result, plan, archive)
    if result.status != "finalized":
        raise RuntimeError(
            f"Server did not finalize the eval import (status {result.status!r})"
        )
    if not result.eval_run_id or not result.eval_run_name or not result.platform_url:
        raise RuntimeError("Finalized eval import response is missing run details")
    if result.uploaded_files != result.expected_files or any(
        file.state != "uploaded" for file in result.files
    ):
        raise RuntimeError("Finalized eval import response has incomplete files")


def _buffered_part(chunks: Iterator[bytes]) -> closing[io.BytesIO]:
    # Closing frees the buffer after its PUT, before the next part is built.
    part = io.BytesIO()
    for chunk in chunks:
        part.write(chunk)
    part.seek(0)
    return closing(part)


def _upload_file_list(
    archive: RunArchive, upload: UploadInfo
) -> list[dict[str, Any]] | None:
    from osmosis_ai.platform.api.upload import (
        _upload_fileobj_multipart,
        _upload_fileobj_simple,
    )

    handle = io.BytesIO(archive.file_list)
    if upload.method == "multipart":
        return _upload_fileobj_multipart(handle, len(archive.file_list), upload)
    _upload_fileobj_simple(handle, len(archive.file_list), upload)
    return None


def _upload_archive(
    plan: EvalUploadPlan, archive: RunArchive, upload: UploadInfo
) -> list[dict[str, Any]]:
    """Stream ``run.zip`` one part at a time, verifying each file as it is read."""
    from osmosis_ai.platform.api.upload import (
        _upload_multipart_parts,
        make_progress_bar,
    )

    if upload.method != "multipart":
        raise RuntimeError(f"Server returned a simple upload for {ARCHIVE_PATH}")
    layout = archive.layout
    console.print(
        f"Uploading {len(plan.files):,} files as {ARCHIVE_PATH} "
        f"({format_size(layout.size)})...",
        style="dim",
    )
    progress, on_progress = make_progress_bar(
        layout.size, description=f"Uploading {ARCHIVE_PATH}"
    )
    with RunFileReader(plan.files) as reader, progress:
        parts = _upload_multipart_parts(
            layout.size,
            upload,
            lambda offset, size: _buffered_part(
                layout.iter_bytes(offset, offset + size, reader.read)
            ),
            on_progress,
        )
        reader.finish()
    return parts


def _upload_slots(
    client: OsmosisClient,
    session_id: str,
    paths: list[str],
    *,
    plan: EvalUploadPlan,
    archive: RunArchive,
    context: GitWorkspaceDirectoryContext | PlatformWorkspaceContext,
) -> None:
    response = client.get_eval_run_import_uploads(
        session_id,
        paths=paths,
        credentials=context.credentials,
        git_identity=context.git_identity,
    )
    instructions = {}
    for instruction in response.files:
        if instruction.path not in paths:
            raise RuntimeError(
                f"Server returned upload instructions for unknown path "
                f"{instruction.path!r}"
            )
        _check_slot(instruction.path, instruction.size, instruction.sha256, archive)
        instructions[instruction.path] = instruction
    for path in (FILE_LIST_PATH, ARCHIVE_PATH):
        instruction = instructions.get(path)
        if instruction is None:
            continue
        parts = (
            _upload_file_list(archive, instruction.upload)
            if path == FILE_LIST_PATH
            else _upload_archive(plan, archive, instruction.upload)
        )
        client.complete_eval_run_import_upload(
            session_id,
            path=path,
            parts=parts,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )


def _wait_for_extraction(
    client: OsmosisClient,
    session_id: str,
    *,
    archive: RunArchive,
    context: GitWorkspaceDirectoryContext | PlatformWorkspaceContext,
) -> None:
    with get_output_context().status("Extracting uploaded files on the platform..."):
        state = client.start_eval_run_import_extraction(
            session_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
        deadline = time.monotonic() + _EXTRACTION_TIMEOUT_SECONDS
        while state.status == "pending":
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "The platform did not finish extracting the uploaded files "
                    f"within {_EXTRACTION_TIMEOUT_SECONDS / 60:.0f} minutes; "
                    "re-run the upload to retry"
                )
            time.sleep(_EXTRACTION_POLL_SECONDS)
            state = client.get_eval_run_import_extraction(
                session_id,
                credentials=context.credentials,
                git_identity=context.git_identity,
            )
    if state.status == "failed":
        raise RuntimeError(
            f"The platform could not extract the uploaded files: {state.reason}. "
            "Re-run the upload to try again"
        )
    if (
        state.file_count != archive.layout.entry_count
        or state.total_bytes != archive.total_bytes
    ):
        raise RuntimeError("Server extracted a different set of files than uploaded")


def prepare_eval_upload_plan(
    run_dir: Path, *, config_path: str | None = None
) -> EvalUploadPlan:
    """Scrub ambient env secrets from logs.txt, then build the import plan."""
    scrub_logs_file(run_dir / LOGS_FILENAME, env=os.environ)
    return build_eval_upload_plan(run_dir, config_path=config_path)


def upload_plan(
    plan: EvalUploadPlan,
    *,
    context: GitWorkspaceDirectoryContext | PlatformWorkspaceContext,
    client: OsmosisClient | None = None,
    replace: bool = False,
) -> EvalRunImportResult:
    """Upload a validated plan. The caller must hold the run's ``RunLock``.

    The run travels as ``files.jsonl`` plus a streamed ``run.zip`` that the
    platform extracts before finalizing. Re-running resumes the same session:
    uploaded slots are skipped and a failed extraction restarts.

    ``replace`` opts into overwriting results this run already imported, which
    is what a ``--retry-failed`` attempt of an uploaded run needs.
    """
    client = client or OsmosisClient()
    archive = build_run_archive(plan.files)
    with get_output_context().status("Starting local evaluation upload..."):
        result = client.start_eval_run_import(
            local_run_id=plan.local_run_id,
            manifest_digest=plan.manifest_digest,
            run=plan.run,
            schema_versions=plan.schema_versions,
            provenance=plan.provenance,
            archive=archive.to_request(),
            reimport=replace,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    _validate_session(result, plan, archive)
    if result.status != "finalized":
        pending = [file.path for file in result.files if file.state == "pending"]
        if pending:
            _upload_slots(
                client,
                result.session_id,
                pending,
                plan=plan,
                archive=archive,
                context=context,
            )
        _wait_for_extraction(
            client, result.session_id, archive=archive, context=context
        )

    with get_output_context().status("Finalizing local evaluation upload..."):
        finalized = client.finalize_eval_run_import(
            result.session_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    _validate_finalized(finalized, plan, archive)
    return finalized


def _result(
    plan: EvalUploadPlan,
    imported: EvalRunImportResult,
    *,
    context: PlatformWorkspaceContext,
) -> OperationResult:
    resource: dict[str, Any] = {
        "session_id": imported.session_id,
        "local_run_id": plan.local_run_id,
        "local_run_path": str(plan.run_dir),
        "eval_run_id": imported.eval_run_id,
        "eval_run_name": imported.eval_run_name,
        "status": imported.status,
        "expected_files": imported.expected_files,
        "uploaded_files": imported.uploaded_files,
        "platform_url": imported.platform_url,
        **workspace_result_context(context),
    }
    name = imported.eval_run_name or plan.run["name"]
    next_steps = (
        [f"View: {imported.platform_url}"]
        if imported.platform_url
        else [f"Inspect the imported run: osmosis eval info {name}"]
    )
    structured = [{"action": "eval_info", "name": name}]
    if imported.platform_url:
        structured.append({"action": "open_url", "url": imported.platform_url})
    return OperationResult(
        operation="eval.upload",
        status="success",
        resource=resource,
        message=f"Uploaded local evaluation: {name}",
        display_next_steps=next_steps,
        next_steps_structured=structured,
    )


def _resolve_run_dir(requested: Path, *, workspace_directory: Path | None) -> Path:
    """Resolve a run name from the default eval root, or preserve an explicit path."""
    requested = requested.expanduser()
    direct = Path(os.path.abspath(requested))
    if requested.is_absolute() or len(requested.parts) != 1:
        return direct
    if direct.is_dir():
        return direct
    try:
        run_name = validate_run_name(str(requested))
    except LocalEvalStateError:
        return direct
    if workspace_directory is None:
        cwd = Path.cwd()
        workspace_directory = find_workspace_directory(cwd) or cwd
    return workspace_directory / ".osmosis" / "evals" / run_name


def upload(run_dir: Path, *, replace: bool = False) -> OperationResult:
    """Upload one completed local evaluation by run name or directory."""
    context = require_platform_workspace_context()
    candidate = _resolve_run_dir(
        run_dir, workspace_directory=context.workspace_directory
    )
    lock_path = candidate.parent / LOCKS_DIRNAME / f"{candidate.name}.lock"
    try:
        with RunLock(lock_path):
            plan = prepare_eval_upload_plan(candidate)
            imported = upload_plan(plan, context=context, replace=replace)
    except LocalEvalUploadError as exc:
        raise CLIError(str(exc)) from exc
    except PlatformAPIError as exc:
        # Only an already-imported run can be republished with --replace; the
        # platform's other conflicts carry their own remedy.
        if exc.error_code == "EVAL_IMPORT_EXISTS" and not replace:
            classified = classify_error(exc)
            scope = (
                f" --workspace {quote(context.workspace_name)}"
                if context.workspace_name
                else ""
            )
            command = f"osmosis{scope} eval upload {quote(str(run_dir))} --replace"
            raise CLIError(
                f"{exc} Re-run it with --retry-failed, then republish with: "
                f"{console.escape(command)}",
                code=classified.code,
                details=classified.details,
            ) from None
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise CLIError(f"Local evaluation upload failed: {exc}") from exc
    return _result(plan, imported, context=context)
