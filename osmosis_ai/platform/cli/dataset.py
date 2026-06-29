"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    CommandResult,
    DetailField,
    DetailResult,
    ListColumn,
    ListResult,
    OperationResult,
    detail_fields,
    get_output_context,
    serialize_dataset,
)
from osmosis_ai.cli.output.display import format_local_date
from osmosis_ai.cli.paths import parse_cli_path
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.api.models import (
    STATUSES_ERROR,
    STATUSES_IN_PROGRESS,
    STATUSES_SUCCESS,
)
from osmosis_ai.platform.auth import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.cli.workspace_directory_context import git_result_context
from osmosis_ai.platform.constants import DEFAULT_PAGE_SIZE

from .constants import (
    MAX_FILE_SIZE,
    MIN_ROW_COUNT,
    REQUIRED_COLUMNS,
    VALID_EXTENSIONS,
)
from .utils import (
    build_dataset_detail_rows,
    build_logs_result,
    format_dataset_status,
    format_size,
    platform_call,
    require_git_workspace_directory_context,
)


def _abort_upload(
    client: Any,
    dataset_id: str,
    *,
    git_identity: str,
    credentials: Any | None = None,
) -> None:
    """Best-effort abort of an in-progress upload."""
    try:
        client.abort_upload(
            dataset_id,
            credentials=credentials,
            git_identity=git_identity,
        )
    except Exception as exc:
        # print_warning (stderr, output-mode aware) rather than console.print,
        # which would write this warning to stdout — polluting piped/redirected
        # output and risking a collision with the live upload progress bar.
        console.print_warning(f"Failed to abort upload: {exc}", code="ABORT_FAILED")


# ── Complete-upload retry (P0 reliability fix) ───────────────────────
# Inspired by huggingface_hub's http_backoff pattern: exponential
# backoff on transient errors (network/timeout/5xx).  This is the most
# dangerous failure point — the file is already on S3, so a failed
# complete call leaves an orphan that costs money and confuses users.
#
# Note: the platform has server-side safety nets for orphaned S3 objects
# (weekly cron Lambda + S3 lifecycle rules), so this retry is the
# primary client-side defense, not the only one.

from osmosis_ai.platform.api.upload import BACKOFF_BASE, BACKOFF_CAP, MAX_RETRIES

PARQUET_VALIDATION_SKIPPED_WARNING = (
    "pyarrow not installed; parquet content validation skipped. "
    "Install with: pip install 'osmosis-ai[platform]'"
)


def _complete_with_retry(
    client: Any,
    dataset_id: str,
    parts: list[dict] | None = None,
    *,
    file_extension: str | None = None,
    git_identity: str,
    credentials: Any | None = None,
) -> Any:
    """Call ``client.complete_upload`` with automatic retry on transient errors.

    On final failure, prints recovery information so the user can retry
    manually with ``osmosis dataset complete <id>``.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
            console.print(
                f"Retrying complete (attempt {attempt} of {MAX_RETRIES - 1})...",
                style="dim",
            )
            time.sleep(delay)
        try:
            return platform_call(
                "Finalizing upload...",
                lambda: client.complete_upload(
                    dataset_id,
                    parts=parts,
                    file_extension=file_extension,
                    credentials=credentials,
                    git_identity=git_identity,
                ),
            )
        except AuthenticationExpiredError:
            raise  # Not transient — surface immediately
        except PlatformAPIError as e:
            # Retry on server errors (5xx), rate limits (429), or no status code (network)
            if (
                e.status_code is not None
                and e.status_code < 500
                and e.status_code != 429
            ):
                raise  # 4xx (except 429) — not transient
            last_error = e
        except Exception as e:
            # Network / timeout / unexpected — retryable
            last_error = e

    # All retries exhausted
    console.print()
    console.print(
        f"Upload succeeded but failed to notify server after {MAX_RETRIES} attempts.",
        style="bold red",
    )
    console.print(f"  Dataset ID: {dataset_id}", style="yellow")
    console.print(
        "  The file may need to be re-uploaded. Please run the upload command again.",
        style="yellow",
    )
    console.print()
    raise CLIError(
        f"Failed to complete upload: {last_error}",
        code="PLATFORM_ERROR",
        details={"dataset_id": dataset_id},
    ) from last_error


def _check_file_basics(file: str) -> tuple[Path, str, int]:
    """Validate file existence, extension, and size.

    Returns (resolved_path, extension, file_size) or raises CLIError.
    """
    file_path = Path(file).resolve()

    try:
        file_size = file_path.stat().st_size
    except FileNotFoundError:
        raise CLIError(f"File not found: {file_path}") from None

    ext = file_path.suffix.lstrip(".").lower()
    if ext not in VALID_EXTENSIONS:
        raise CLIError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
    if file_size == 0:
        raise CLIError("File is empty.")
    if file_size > MAX_FILE_SIZE:
        raise CLIError(
            f"File too large ({format_size(file_size)}). Maximum: {format_size(MAX_FILE_SIZE)}"
        )
    return file_path, ext, file_size


def _dataset_name_from_path(file_path: Path) -> str:
    """Derive the platform dataset name from a local file path."""
    return file_path.stem


def _existing_dataset_id_from_conflict(exc: PlatformAPIError) -> str | None:
    """Extract the recoverable duplicate-name conflict marker from a platform error."""
    if exc.status_code != 409 or not isinstance(exc.details, dict):
        return None
    existing_dataset_id = exc.details.get("existing_dataset_id")
    return existing_dataset_id if isinstance(existing_dataset_id, str) else None


def _create_dataset_for_upload(
    *,
    client: Any,
    dataset_name: str,
    file_size: int,
    ext: str,
    overwrite: bool,
    git_identity: str,
    credentials: Any | None = None,
) -> Any:
    """Create the dataset record, retrying with overwrite when explicitly requested."""
    try:
        return platform_call(
            "Uploading dataset...",
            lambda: client.create_dataset(
                dataset_name,
                file_size,
                ext,
                credentials=credentials,
                git_identity=git_identity,
            ),
        )
    except PlatformAPIError as exc:
        existing_dataset_id = _existing_dataset_id_from_conflict(exc)
        if existing_dataset_id is None:
            raise
        if not overwrite:
            details = dict(exc.details or {})
            details.setdefault("status_code", exc.status_code)
            raise CLIError(
                f"A dataset named '{dataset_name}' already exists. "
                "Use --overwrite to replace it.",
                code="CONFLICT",
                details=details,
            ) from exc

        return platform_call(
            "Replacing existing dataset...",
            lambda: client.create_dataset(
                dataset_name,
                file_size,
                ext,
                overwrite_dataset_id=existing_dataset_id,
                credentials=credentials,
                git_identity=git_identity,
            ),
        )


def _perform_upload(
    *,
    file_path: Path,
    ext: str,
    file_size: int,
    git_identity: str,
    credentials: Any | None = None,
    overwrite: bool = False,
) -> Any:
    """Core upload: create dataset record → S3 upload → complete.

    Returns the completed DatasetFile. Raises CLIError on failure.
    Shared by the ``upload`` CLI command and the workspace interactive flow.
    """
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.upload import (
        make_progress_bar,
        upload_file_multipart,
        upload_file_simple,
    )

    client = OsmosisClient()
    dataset_name = _dataset_name_from_path(file_path)

    dataset = _create_dataset_for_upload(
        client=client,
        dataset_name=dataset_name,
        file_size=file_size,
        ext=ext,
        overwrite=overwrite,
        credentials=credentials,
        git_identity=git_identity,
    )

    upload_info = dataset.upload
    if upload_info is None:
        raise CLIError("Server did not return upload instructions.")

    is_multipart = upload_info.method == "multipart"
    if is_multipart:
        parts_label = (
            f", {upload_info.total_parts} parts" if upload_info.total_parts else ""
        )
        console.print(
            f"Uploading {console.escape(file_path.name)} "
            f"({format_size(file_size)}, multipart{parts_label})..."
        )
    else:
        console.print(
            f"Uploading {console.escape(file_path.name)} ({format_size(file_size)})..."
        )

    ctx, progress_cb = make_progress_bar(file_size)

    if is_multipart:
        try:
            with ctx:
                parts = upload_file_multipart(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                git_identity=git_identity,
            )
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                git_identity=git_identity,
            )
            raise CLIError(f"Upload failed: {e}") from e
        completed = _complete_with_retry(
            client,
            dataset.id,
            parts=parts,
            file_extension=ext,
            credentials=credentials,
            git_identity=git_identity,
        )
    else:
        try:
            with ctx:
                upload_file_simple(
                    file_path, upload_info, progress_callback=progress_cb
                )
        except KeyboardInterrupt:
            console.print("\nUpload interrupted.")
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                git_identity=git_identity,
            )
            raise CLIError("Upload cancelled by user.") from None
        except Exception as e:
            _abort_upload(
                client,
                dataset.id,
                credentials=credentials,
                git_identity=git_identity,
            )
            raise CLIError(f"Upload failed: {e}") from e
        completed = _complete_with_retry(
            client,
            dataset.id,
            file_extension=ext,
            credentials=credentials,
            git_identity=git_identity,
        )

    return completed or dataset


def upload(
    file: str,
    overwrite: bool = False,
    yes: bool = False,
) -> CommandResult:
    """Upload a dataset file."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity

    file_path, ext, file_size = _check_file_basics(file)

    # Validate file contents before uploading
    errors = _validate_file(file_path, ext)
    if errors:
        raise CLIError(
            "File validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    overwrite_warning = (
        "--overwrite will replace the existing dataset with the same name."
    )
    require_confirmation(
        (
            "This will overwrite the existing dataset with the same name. "
            "Proceed with upload?"
            if overwrite
            else "Proceed with upload?"
        ),
        yes=yes,
        # Default to "no" for the destructive overwrite path so a bare Enter
        # does not replace data; non-destructive uploads keep the "yes" default.
        default=not overwrite,
        summary=[("File", file_path.name), ("Size", format_size(file_size))],
        warnings=[overwrite_warning] if overwrite else None,
    )

    dataset = _perform_upload(
        file_path=file_path,
        ext=ext,
        file_size=file_size,
        overwrite=overwrite,
        credentials=credentials,
        git_identity=git_identity,
    )

    resource = serialize_dataset(dataset)
    resource.update(git_result_context(context))
    display_next_steps = [
        (
            f"Processing will continue on the platform. Check status at: {dataset.platform_url}"
            if dataset.platform_url
            else f"Processing will continue on the platform. Check status with: osmosis dataset info {dataset.file_name}"
        )
    ]
    next_steps_structured = (
        [{"label": "View dataset", "url": dataset.platform_url}]
        if dataset.platform_url
        else [{"action": "dataset_info", "id": dataset.id}]
    )
    return OperationResult(
        operation="dataset.upload",
        status="success",
        resource=resource,
        message=f"Dataset uploaded: {dataset.file_name}",
        display_next_steps=display_next_steps,
        next_steps_structured=next_steps_structured,
    )


def list_datasets(limit: int = DEFAULT_PAGE_SIZE, all_: bool = False) -> CommandResult:
    """List datasets."""
    from osmosis_ai.platform.cli.utils import (
        paginated_fetch,
        validate_list_options,
    )

    effective_limit, fetch_all = validate_list_options(limit=limit, all_=all_)

    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()

    with get_output_context().status("Fetching datasets..."):
        datasets, total_count, has_more, next_offset = paginated_fetch(
            lambda lim, off: client.list_datasets(
                limit=lim,
                offset=off,
                credentials=credentials,
                git_identity=git_identity,
            ),
            items_attr="datasets",
            limit=effective_limit,
            fetch_all=fetch_all,
        )

    return ListResult(
        title="Datasets",
        items=[serialize_dataset(d) for d in datasets],
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
        extra=git_result_context(context),
        columns=[
            ListColumn(key="file_name", label="Name", ratio=4, overflow="fold"),
            ListColumn(key="status", label="Status", no_wrap=True, ratio=1),
            ListColumn(key="file_size", label="Size", no_wrap=True, ratio=1),
            ListColumn(key="created_at", label="Uploaded", no_wrap=True, ratio=1),
            ListColumn(key="creator_name", label="Uploaded By", no_wrap=True, ratio=1),
        ],
        display_items=[
            {
                **serialize_dataset(d),
                "status": format_dataset_status(d),
                "file_size": format_size(d.file_size),
                "created_at": format_local_date(d.created_at),
                "creator_name": d.creator_name or "–",
            }
            for d in datasets
        ],
    )


def info(
    name: str,
) -> CommandResult:
    """Show dataset details and processing status."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            git_identity=git_identity,
        ),
    )

    rows = build_dataset_detail_rows(ds, include_id=ds.is_internal_user)
    data = serialize_dataset(ds)
    data.update(git_result_context(context))
    display_hints = [f"View: {ds.platform_url}"] if ds.platform_url else []
    if ds.status in STATUSES_ERROR:
        display_hints.append(
            f"See logs with: osmosis dataset logs {ds.file_name or name}"
        )
    return DetailResult(
        title="Dataset",
        data=data,
        fields=detail_fields(rows),
        display_hints=display_hints,
    )


def logs(name: str, *, limit: int, cursor: str | None = None) -> ListResult:
    """Show the most recent logs for a dataset, oldest-first."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    page = platform_call(
        "Fetching logs...",
        lambda: client.get_dataset_logs(
            name,
            limit=limit,
            cursor=cursor,
            credentials=credentials,
            git_identity=git_identity,
        ),
    )

    return build_logs_result(
        title=f"Dataset Logs: {name}",
        page=page,
        context=context,
        next_step_hint=f"Use osmosis dataset info {name} for dataset details.",
    )


def preview(
    name: str,
    rows: int = 5,
) -> CommandResult:
    """Preview dataset rows."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    from osmosis_ai.platform.api.client import OsmosisClient

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            git_identity=git_identity,
        ),
    )

    if ds.data_preview is None:
        if ds.status in STATUSES_IN_PROGRESS:
            message = f"Dataset is still processing (status: {ds.status})."
        else:
            message = "No preview available for this dataset."
        return DetailResult(
            title="Dataset Preview",
            data={
                "dataset": serialize_dataset(ds),
                "rows": None,
                "requested_rows": rows,
                "returned_rows": 0,
                "available": False,
                "message": message,
                **git_result_context(context),
            },
            fields=[
                DetailField(label="Dataset", value=ds.file_name),
                DetailField(label="Status", value=ds.status),
                DetailField(label="Preview", value=message),
            ],
        )

    data_rows = ds.data_preview
    if isinstance(data_rows, list):
        limit = min(rows, len(data_rows))
        preview_rows = data_rows[:limit]
        fields = [
            DetailField(label="Dataset", value=ds.file_name),
            DetailField(label="Rows", value=f"{len(preview_rows)} of {len(data_rows)}"),
        ]
        fields.extend(
            DetailField(label=f"Row {idx}", value=str(row))
            for idx, row in enumerate(preview_rows, 1)
        )
    else:
        preview_rows = data_rows
        fields = [
            DetailField(label="Dataset", value=ds.file_name),
            DetailField(label="Preview", value=str(data_rows)),
        ]

    return DetailResult(
        title="Dataset Preview",
        data={
            "dataset": serialize_dataset(ds),
            "rows": preview_rows,
            "requested_rows": rows,
            "returned_rows": len(preview_rows) if isinstance(preview_rows, list) else 1,
            "available": True,
            **git_result_context(context),
        },
        fields=fields,
    )


def _default_download_filename(file_name: str, presigned_url: str) -> str:
    """Choose a useful filename when the platform response omits one."""
    if Path(file_name).suffix:
        return file_name

    suffix = PurePosixPath(unquote(urlparse(presigned_url).path)).suffix
    if suffix.lstrip(".").lower() in VALID_EXTENSIONS:
        return f"{file_name}{suffix}"
    return file_name


def download(
    name: str,
    output: str | None = None,
    overwrite: bool = False,
) -> CommandResult:
    """Download a dataset file."""
    context = require_git_workspace_directory_context()
    credentials = context.credentials
    git_identity = context.git_identity
    from osmosis_ai.platform.api.client import OsmosisClient
    from osmosis_ai.platform.api.download import download_file

    client = OsmosisClient()
    ds = platform_call(
        "Fetching dataset...",
        lambda: client.get_dataset(
            name,
            credentials=credentials,
            git_identity=git_identity,
        ),
    )
    if ds.status not in STATUSES_SUCCESS:
        if ds.status in STATUSES_IN_PROGRESS:
            raise CLIError(
                f"Dataset is still processing (status: {ds.status}). "
                "Try again after it finishes uploading."
            )
        raise CLIError(f"Dataset is not available for download (status: {ds.status}).")

    info = platform_call(
        "Preparing download...",
        lambda: client.get_dataset_download_url(
            name,
            credentials=credentials,
            git_identity=git_identity,
        ),
    )
    default_filename = _default_download_filename(
        info.file_name or ds.file_name,
        info.presigned_url,
    )
    parsed_output = parse_cli_path(output, expand_user=True) if output else None
    output_path = parsed_output.path if parsed_output else None

    try:
        destination = download_file(
            info.presigned_url,
            output=output_path,
            default_filename=default_filename,
            expected_size=ds.file_size,
            overwrite=overwrite,
            output_is_directory=(
                parsed_output.has_trailing_separator if parsed_output else False
            ),
        )
    except FileExistsError as exc:
        raise CLIError(f"{exc} Use --overwrite to replace it.") from None
    except Exception as exc:
        raise CLIError(f"Download failed: {exc}") from exc

    resource = serialize_dataset(ds)
    resource["output_path"] = str(destination)
    resource.update(git_result_context(context))
    return OperationResult(
        operation="dataset.download",
        status="success",
        resource=resource,
        message=f"Dataset downloaded: {destination}",
    )


def validate(
    file: str,
) -> CommandResult:
    """Validate a dataset file locally."""
    file_path, ext, file_size = _check_file_basics(file)

    # Format validation
    errors, warnings = _validate_file_with_warnings(file_path, ext)

    if errors:
        raise CLIError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return DetailResult(
        title="Dataset Validation",
        data={
            "valid": True,
            "file": str(file_path),
            "file_name": file_path.name,
            "extension": ext,
            "file_size": file_size,
            "errors": [],
            "warnings": warnings,
        },
        fields=[
            DetailField(label="File", value=file_path.name),
            DetailField(label="Size", value=format_size(file_size)),
            DetailField(label="Status", value=f"Valid {ext} file"),
            *[DetailField(label="Warning", value=warning) for warning in warnings],
        ],
    )


def _validate_file_with_warnings(
    file_path: Path, ext: str
) -> tuple[list[str], list[str]]:
    """Validate file contents and surface non-fatal validation skips."""
    warnings: list[str] = []
    if ext == "parquet":
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            warnings.append(PARQUET_VALIDATION_SKIPPED_WARNING)
            return [], warnings
    return _validate_file(file_path, ext), warnings


def _validate_file(file_path: Path, ext: str) -> list[str]:
    """Validate file contents based on extension."""
    if ext == "jsonl":
        return _validate_jsonl(file_path)
    elif ext == "csv":
        return _validate_csv(file_path)
    elif ext == "parquet":
        return _validate_parquet(file_path)
    return []


def _check_required_columns(columns: Iterable[str]) -> list[str]:
    """Check that required columns are present."""
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        return [
            f"Missing required columns: {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(REQUIRED_COLUMNS))}"
        ]
    return []


# ── Optional "metadata" column validation ─────────────────────────
# Datasets may carry an optional per-row "metadata" column. Its
# canonical form is a JSON object (dict). Users may author it as a
# native object (JSONL/Parquet) or as a JSON-object string (CSV;
# tolerated in JSONL). String->object normalization happens only in
# the platform; the SDK mirrors the platform's user-error rules over
# the sampled rows for early feedback: a cell must be a JSON object
# without empty nested objects, per-key value types must agree across
# rows, and the sampled objects must not all be empty ({}).

METADATA_COLUMN = "metadata"


def _metadata_is_absent(value: Any) -> bool:
    """Return True when a metadata cell should be treated as absent.

    None and empty / whitespace-only strings are all "absent" and skip
    further shape validation.
    """
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _contains_nested_empty_object(value: Any, *, is_root: bool = True) -> bool:
    """Return whether a metadata value contains a nested empty object."""
    if isinstance(value, dict):
        if not value:
            return not is_root
        return any(
            _contains_nested_empty_object(child, is_root=False)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _contains_nested_empty_object(child, is_root=False) for child in value
        )
    return False


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _contains_oversized_int(value: Any) -> bool:
    """Return whether a metadata value contains an int beyond 64 bits.

    Valid JSON, but the platform stores metadata as a parquet struct and
    Arrow cannot hold an integer outside the int64 range.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return not _INT64_MIN <= value <= _INT64_MAX
    if isinstance(value, dict):
        return any(_contains_oversized_int(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_oversized_int(child) for child in value)
    return False


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    # JSON has a single "number" type: int and float for the same key are
    # consistent (the platform promotes them to double).
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


class _MetadataCrossRowTracker:
    """Cross-row metadata checks over the sampled rows.

    Mirrors the platform normalizer's user-error rules that cannot be
    checked per cell: per-key JSON value types must agree across rows, and
    metadata objects must not all be empty ({}) — an all-empty column has
    no keys and cannot be stored as a parquet struct.
    """

    def __init__(self) -> None:
        self._types: dict[str, str] = {}
        self._saw_empty_object = False
        self._saw_keyed_object = False

    def observe(self, value: dict, *, location: str) -> list[str]:
        """Record one metadata object; return cross-row type errors."""
        if value:
            self._saw_keyed_object = True
        else:
            self._saw_empty_object = True
        return self._collect(value, "$", location)

    def _collect(self, value: Any, path: str, location: str) -> list[str]:
        if value is None:
            return []
        current = _json_type_name(value)
        previous = self._types.setdefault(path, current)
        if previous != current:
            return [
                f"{location}: invalid metadata - value type at {path} is "
                f"inconsistent across rows ({previous} vs {current})"
            ]
        errors: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                errors.extend(self._collect(child, f"{path}.{key}", location))
        elif isinstance(value, list):
            for child in value:
                errors.extend(self._collect(child, f"{path}[]", location))
        return errors

    def finish(self) -> list[str]:
        """Return errors only determinable after all sampled rows are seen."""
        if self._saw_empty_object and not self._saw_keyed_object:
            return [
                "Invalid metadata column: all sampled metadata objects are "
                "empty ({}); they carry no keys and cannot be stored. Remove "
                "the metadata column or add at least one key."
            ]
        return []


def _check_metadata_value(
    value: Any,
    *,
    location: str,
    tracker: _MetadataCrossRowTracker | None = None,
) -> list[str]:
    """Validate a single metadata cell, returning errors (empty if valid).

    A cell must be a dict, or a string that JSON-parses to an object (dict),
    without empty nested objects. When a ``tracker`` is given the object is
    also recorded for cross-row consistency checks. ``location`` is a
    human-readable prefix such as ``"Line 3"`` or ``"Near end of file"``.
    """
    import json

    if _metadata_is_absent(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            return [f"{location}: invalid metadata - not valid JSON ({e})"]
    if not isinstance(value, dict):
        return [
            f"{location}: invalid metadata - must be a JSON object, "
            f"got {type(value).__name__}"
        ]
    if _contains_nested_empty_object(value):
        return [
            f"{location}: invalid metadata - contains an empty nested "
            "object ({}), which cannot be stored. Remove that key or add "
            "at least one nested key."
        ]
    if _contains_oversized_int(value):
        return [
            f"{location}: invalid metadata - contains an integer too large "
            "to store (must fit in 64 bits)"
        ]
    if tracker is not None:
        return tracker.observe(value, location=location)
    return []


def _read_tail_lines(
    file_path: Path, n: int, chunk_size: int = 1024 * 1024
) -> list[str]:
    """Read approximately the last *n* lines of a text file.

    Seeks to the end and reads a chunk backwards. If the file is smaller
    than *chunk_size* the entire content is returned (caller should handle
    overlap with head validation).
    """
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return []
        read_size = min(chunk_size, file_size)
        f.seek(file_size - read_size)
        data = f.read(read_size)

    # If we read a chunk that doesn't start at the beginning of the file,
    # it may begin mid-way through a multi-byte UTF-8 character.  Strip
    # leading continuation bytes (10xxxxxx, i.e. 0x80-0xBF) so the
    # decode doesn't fail on a perfectly valid UTF-8 file.
    if read_size < file_size:
        start = 0
        while start < len(data) and 0x80 <= data[start] <= 0xBF:
            start += 1
        data = data[start:]

    text = data.decode("utf-8")

    lines = text.split("\n")
    # If we didn't read from the start, the first "line" may be partial
    if read_size < file_size:
        lines = lines[1:]
    # Drop trailing empty string produced by a final newline
    if lines and not lines[-1]:
        lines = lines[:-1]
    return lines[-n:]


def _check_parquet_metadata_column(pf: Any) -> list[str]:
    """Validate the optional "metadata" column of a parquet file.

    A struct dtype is accepted as-is: parquet enforces one uniform struct
    schema per column and cannot store empty structs, so the cross-row
    user-error rules cannot be violated. A string / large_string dtype is
    accepted only when every cell in the first batch parses to a JSON
    object (absent cells are skipped). Any other dtype is rejected.
    """
    import pyarrow as pa

    schema = pf.schema_arrow
    if METADATA_COLUMN not in schema.names:
        return []

    field_type = schema.field(METADATA_COLUMN).type
    # A struct column is already an object; an all-null column means every
    # cell is absent. Both are accepted without per-cell inspection.
    if pa.types.is_struct(field_type) or pa.types.is_null(field_type):
        return []
    if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
        return [
            f"Invalid metadata column: must be a struct or JSON-object string, "
            f"got dtype {field_type}"
        ]

    # String dtype: validate parseability on the first batch.
    errors: list[str] = []
    tracker = _MetadataCrossRowTracker()
    for batch in pf.iter_batches(columns=[METADATA_COLUMN]):
        for value in batch.column(0).to_pylist():
            errors.extend(
                _check_metadata_value(
                    value, location="metadata column", tracker=tracker
                )
            )
            if len(errors) >= 5:
                errors.append("... (showing first 5 errors)")
                return errors
        break
    errors.extend(tracker.finish())
    return errors


def _validate_parquet(file_path: Path) -> list[str]:
    """Validate parquet file structure and required columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        console.print(
            f"Note: {PARQUET_VALIDATION_SKIPPED_WARNING}",
            style="dim yellow",
        )
        return []

    errors = []
    try:
        with pq.ParquetFile(file_path) as pf:
            if len(pf.schema) == 0:
                errors.append("Parquet file has no columns")
            elif pf.metadata.num_rows == 0:
                errors.append("Parquet file has no rows")
            else:
                errors.extend(_check_required_columns(pf.schema_arrow.names))
                if pf.metadata.num_rows < MIN_ROW_COUNT:
                    errors.append(
                        f"Dataset too small: {pf.metadata.num_rows} rows. "
                        f"A minimum of {MIN_ROW_COUNT} rows is required."
                    )
                errors.extend(_check_parquet_metadata_column(pf))
    except Exception as e:
        errors.append(f"Invalid parquet file: {e}")
    return errors


def _validate_jsonl(file_path: Path) -> list[str]:
    """Validate JSONL: required columns + first/last 100 lines."""
    import json

    errors = []
    columns_checked = False
    file_fully_read = False
    tracker = _MetadataCrossRowTracker()

    row_count = 0
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 100:
                break
            stripped = line.strip()
            if not stripped:
                continue
            row_count += 1
            try:
                obj = json.loads(stripped)
                if not columns_checked and isinstance(obj, dict):
                    errors.extend(_check_required_columns(obj.keys()))
                    columns_checked = True
                if isinstance(obj, dict) and METADATA_COLUMN in obj:
                    errors.extend(
                        _check_metadata_value(
                            obj[METADATA_COLUMN],
                            location=f"Line {i}",
                            tracker=tracker,
                        )
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        return errors
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    return errors
        else:
            # Loop completed without break — entire file was read
            file_fully_read = True

    if file_fully_read:
        errors.extend(tracker.finish())
        if row_count < MIN_ROW_COUNT:
            errors.append(
                f"Dataset too small: {row_count} rows. "
                f"A minimum of {MIN_ROW_COUNT} rows is required."
            )
        return errors

    if len(errors) >= 5:
        return errors

    # Validate last 100 lines
    try:
        tail_lines = _read_tail_lines(file_path, 100)
    except UnicodeDecodeError:
        errors.append("File contains invalid UTF-8 encoding near end of file")
        return errors
    for line in tail_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if not columns_checked and isinstance(obj, dict):
                errors.extend(_check_required_columns(obj.keys()))
                columns_checked = True
            if isinstance(obj, dict) and METADATA_COLUMN in obj:
                errors.extend(
                    _check_metadata_value(
                        obj[METADATA_COLUMN],
                        location="Near end of file",
                        tracker=tracker,
                    )
                )
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    break
        except json.JSONDecodeError as e:
            errors.append(f"Near end of file: invalid JSON - {e}")
            if len(errors) >= 5:
                errors.append("... (showing first 5 errors)")
                break

    if len(errors) < 5:
        errors.extend(tracker.finish())

    if row_count < MIN_ROW_COUNT:
        total = 0
        with open(file_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    total += 1
                    if total >= MIN_ROW_COUNT:
                        break
        if total < MIN_ROW_COUNT:
            errors.append(
                f"Dataset too small: {total} rows. "
                f"A minimum of {MIN_ROW_COUNT} rows is required."
            )

    return errors


def _validate_csv(file_path: Path) -> list[str]:
    """Validate CSV: required columns + first/last 100 rows."""
    import csv

    errors = []
    num_cols = 0
    row_count = 0
    file_fully_read = False
    tracker = _MetadataCrossRowTracker()

    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return ["File has no header row"]

            errors.extend(_check_required_columns(header))
            num_cols = len(header)
            metadata_idx = (
                header.index(METADATA_COLUMN) if METADATA_COLUMN in header else None
            )

            for i, row in enumerate(reader, 2):
                if i > 101:
                    break
                row_count += 1
                if len(row) != num_cols:
                    errors.append(
                        f"Row {i}: expected {num_cols} columns, got {len(row)}"
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        return errors
                    continue
                if metadata_idx is not None:
                    errors.extend(
                        _check_metadata_value(
                            row[metadata_idx], location=f"Row {i}", tracker=tracker
                        )
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        return errors
            else:
                file_fully_read = True
    except UnicodeDecodeError as e:
        errors.append(f"File encoding error: {e}")
        return errors
    except csv.Error as e:
        errors.append(f"CSV parse error: {e}")
        return errors

    if file_fully_read:
        errors.extend(tracker.finish())
        if row_count < MIN_ROW_COUNT:
            errors.append(
                f"Dataset too small: {row_count} rows. "
                f"A minimum of {MIN_ROW_COUNT} rows is required."
            )
        return errors

    if len(errors) >= 5:
        return errors

    # Validate last 100 rows
    try:
        tail_lines = _read_tail_lines(file_path, 100)
    except UnicodeDecodeError:
        errors.append("File contains invalid UTF-8 encoding near end of file")
        return errors
    try:
        reader = csv.reader(tail_lines)
        for row in reader:
            if len(row) != num_cols:
                errors.append(
                    f"Near end of file: expected {num_cols} columns, got {len(row)}"
                )
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    break
                continue
            if metadata_idx is not None:
                errors.extend(
                    _check_metadata_value(
                        row[metadata_idx],
                        location="Near end of file",
                        tracker=tracker,
                    )
                )
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    break
    except csv.Error as e:
        errors.append(f"Near end of file: CSV parse error - {e}")

    if len(errors) < 5:
        errors.extend(tracker.finish())

    return errors
