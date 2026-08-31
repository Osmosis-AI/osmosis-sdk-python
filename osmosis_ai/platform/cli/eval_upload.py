"""Handler for ``osmosis eval upload`` and ``eval run --upload``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.eval.local.state import (
    LOCKS_DIRNAME,
    LocalEvalStateError,
    RunLock,
    validate_run_name,
)
from osmosis_ai.eval.local.upload import (
    EvalUploadFile,
    EvalUploadPlan,
    LocalEvalUploadError,
    build_eval_upload_plan,
)
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import EvalRunImportResult
from osmosis_ai.platform.cli.utils import require_git_workspace_directory_context
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context

if TYPE_CHECKING:
    from osmosis_ai.platform.cli.workspace_directory_context import (
        GitWorkspaceDirectoryContext,
    )


def _file_map(plan: EvalUploadPlan) -> dict[str, EvalUploadFile]:
    return {file.path: file for file in plan.files}


def _validate_result_files(
    result: EvalRunImportResult, files: dict[str, EvalUploadFile]
) -> None:
    returned = {file.path: file for file in result.files}
    if set(returned) != set(files):
        missing = sorted(set(files) - set(returned))
        unknown = sorted(set(returned) - set(files))
        detail = []
        if missing:
            detail.append(f"missing {missing[:5]}")
        if unknown:
            detail.append(f"unknown {unknown[:5]}")
        raise RuntimeError(
            "Server returned a different eval import file set"
            + (f": {'; '.join(detail)}" if detail else "")
        )
    for path, returned_file in returned.items():
        local = files[path]
        if returned_file.size != local.size or returned_file.sha256 != local.sha256:
            raise RuntimeError(
                f"Server returned mismatched size or sha256 for {path!r}"
            )
    uploaded = sum(file.state == "uploaded" for file in result.files)
    if result.expected_files != len(files) or result.uploaded_files != uploaded:
        raise RuntimeError("Server returned inconsistent eval import file counts")


def _validate_finalized(result: EvalRunImportResult) -> None:
    if result.status != "finalized":
        raise RuntimeError(
            f"Server did not finalize the eval import (status {result.status!r})"
        )
    if not result.eval_run_id or not result.eval_run_name or not result.platform_url:
        raise RuntimeError("Finalized eval import response is missing run details")
    if (
        result.uploaded_files != result.expected_files
        or len(result.files) != result.expected_files
        or any(file.state != "uploaded" for file in result.files)
    ):
        raise RuntimeError("Finalized eval import response has incomplete files")


def _upload_one(
    *,
    client: OsmosisClient,
    session_id: str,
    file: EvalUploadFile,
    upload: Any,
    context: GitWorkspaceDirectoryContext,
) -> None:
    from osmosis_ai.platform.api.upload import (
        _upload_fileobj_multipart,
        _upload_fileobj_simple,
    )

    with file.open_verified() as handle:
        if upload.method == "multipart":
            parts = _upload_fileobj_multipart(handle, file.size, upload)
        else:
            _upload_fileobj_simple(handle, file.size, upload)
            parts = None
    client.complete_eval_run_import_upload(
        session_id,
        path=file.path,
        parts=parts,
        credentials=context.credentials,
        git_identity=context.git_identity,
    )


def upload_plan(
    plan: EvalUploadPlan,
    *,
    context: GitWorkspaceDirectoryContext,
    client: OsmosisClient | None = None,
) -> EvalRunImportResult:
    """Upload a validated plan. The caller must hold the run's ``RunLock``."""
    client = client or OsmosisClient()
    files = _file_map(plan)
    with get_output_context().status("Starting local evaluation upload..."):
        result = client.start_eval_run_import(
            local_run_id=plan.local_run_id,
            manifest_digest=plan.manifest_digest,
            run=plan.run,
            schema_versions=plan.schema_versions,
            provenance=plan.provenance,
            files=plan.file_requests(),
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    _validate_result_files(result, files)
    if result.status != "finalized":
        for offset in range(0, len(plan.files), 100):
            batch = plan.files[offset : offset + 100]
            requested = {file.path for file in batch}
            response = client.get_eval_run_import_uploads(
                result.session_id,
                paths=[file.path for file in batch],
                credentials=context.credentials,
                git_identity=context.git_identity,
            )
            for instruction in response.files:
                if instruction.path not in requested or instruction.path not in files:
                    raise RuntimeError(
                        f"Server returned upload instructions for unknown path "
                        f"{instruction.path!r}"
                    )
                local = files[instruction.path]
                if instruction.size != local.size or instruction.sha256 != local.sha256:
                    raise RuntimeError(
                        f"Server returned mismatched size or sha256 for "
                        f"{instruction.path!r}"
                    )
                console.print(
                    f"Uploading {console.escape(instruction.path)}...", style="dim"
                )
                _upload_one(
                    client=client,
                    session_id=result.session_id,
                    file=local,
                    upload=instruction.upload,
                    context=context,
                )

    with get_output_context().status("Finalizing local evaluation upload..."):
        finalized = client.finalize_eval_run_import(
            result.session_id,
            credentials=context.credentials,
            git_identity=context.git_identity,
        )
    _validate_result_files(finalized, files)
    _validate_finalized(finalized)
    return finalized


def _result(
    plan: EvalUploadPlan,
    imported: EvalRunImportResult,
    *,
    context: GitWorkspaceDirectoryContext,
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
        **git_result_context(context),
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


def _resolve_run_dir(requested: Path, *, workspace_directory: Path) -> Path:
    """Resolve a run name from the default eval root, or preserve an explicit path."""
    requested = requested.expanduser()
    direct = Path(os.path.abspath(requested))
    if requested.is_absolute() or len(requested.parts) != 1:
        return direct
    try:
        run_name = validate_run_name(str(requested))
    except LocalEvalStateError:
        return direct
    return Path(os.path.abspath(workspace_directory / ".osmosis" / "evals" / run_name))


def upload(run_dir: Path) -> OperationResult:
    """Upload one completed local evaluation by run name or directory."""
    context = require_git_workspace_directory_context()
    candidate = _resolve_run_dir(
        run_dir, workspace_directory=context.workspace_directory
    )
    lock_path = candidate.parent / LOCKS_DIRNAME / f"{candidate.name}.lock"
    try:
        with RunLock(lock_path):
            plan = build_eval_upload_plan(candidate)
            imported = upload_plan(plan, context=context)
    except LocalEvalUploadError as exc:
        raise CLIError(str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise CLIError(f"Local evaluation upload failed: {exc}") from exc
    return _result(plan, imported, context=context)
