"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Iterable
from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import PlatformAPIError

from .project import _require_auth, _require_subscription, _resolve_project

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
REQUIRED_COLUMNS = {"system_prompt", "user_prompt", "ground_truth"}


def _format_size(size_bytes: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return (
                f"{size_bytes:.1f} {unit}"
                if unit != "B"
                else f"{int(size_bytes)} {unit}"
            )
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _resolve_project_id(args: argparse.Namespace) -> str:
    """Get project ID from --project arg, env, or default."""
    project_name = getattr(args, "project", None)
    project = _resolve_project(project_name)
    return project["id"]


class DatasetCommand:
    """Handler for `osmosis dataset`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest="dataset_action", help="Dataset management commands"
        )

        # dataset upload
        upload_parser = subparsers.add_parser("upload", help="Upload a dataset file")
        upload_parser.add_argument("file", help="Path to the file to upload")
        upload_parser.add_argument(
            "--project", help="Project name (default: current project)"
        )
        upload_parser.set_defaults(handler=self._run_upload)

        # dataset list
        list_parser = subparsers.add_parser("list", help="List datasets")
        list_parser.add_argument(
            "--project", help="Project name (default: current project)"
        )
        list_parser.set_defaults(handler=self._run_list)

        # dataset status
        status_parser = subparsers.add_parser(
            "status", help="Check dataset processing status"
        )
        status_parser.add_argument("id", help="Dataset ID")
        status_parser.set_defaults(handler=self._run_status)

        # dataset preview
        preview_parser = subparsers.add_parser("preview", help="Preview dataset rows")
        preview_parser.add_argument("id", help="Dataset ID")
        preview_parser.add_argument(
            "--rows", type=int, default=5, help="Number of rows to show"
        )
        preview_parser.set_defaults(handler=self._run_preview)

        # dataset delete
        delete_parser = subparsers.add_parser("delete", help="Delete a dataset")
        delete_parser.add_argument("id", help="Dataset ID")
        delete_parser.add_argument(
            "-y", "--yes", action="store_true", help="Skip confirmation"
        )
        delete_parser.set_defaults(handler=self._run_delete)

        # dataset validate
        validate_parser = subparsers.add_parser(
            "validate", help="Validate a dataset file locally"
        )
        validate_parser.add_argument("file", help="Path to the file to validate")
        validate_parser.set_defaults(handler=self._run_validate)

        parser.set_defaults(handler=self._run_default)

    @staticmethod
    def _abort_multipart(client, dataset_id: str, upload_id: str | None) -> None:
        """Best-effort abort of a multipart upload."""
        if upload_id:
            with contextlib.suppress(Exception):
                client.abort_upload(dataset_id, upload_id)

    def _run_default(self, args: argparse.Namespace) -> int:
        print("Usage: osmosis dataset <command>")
        print("")
        print("Commands:")
        print("  upload    Upload a dataset file")
        print("  list      List datasets in a project")
        print("  status    Check dataset processing status")
        print("  preview   Preview dataset rows")
        print("  delete    Delete a dataset")
        print("  validate  Validate a dataset file locally")
        return 0

    def _run_upload(self, args: argparse.Namespace) -> int:
        _require_auth()
        _require_subscription()

        from osmosis_ai.platform.api.client import OsmosisClient
        from osmosis_ai.platform.api.upload import (
            make_progress_bar,
            upload_file_multipart,
            upload_file_simple,
        )

        file_path = Path(args.file).resolve()
        if not file_path.exists():
            raise CLIError(f"File not found: {file_path}")

        ext = file_path.suffix.lstrip(".").lower()
        if ext not in VALID_EXTENSIONS:
            raise CLIError(
                f"Unsupported file type '.{ext}'. "
                f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
            )

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise CLIError(
                f"File too large ({_format_size(file_size)}). Maximum: {_format_size(MAX_FILE_SIZE)}"
            )
        if file_size == 0:
            raise CLIError("File is empty.")

        # Validate file contents before uploading
        errors = _validate_file(file_path, ext)
        if errors:
            raise CLIError(
                "File validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        project_id = _resolve_project_id(args)
        client = OsmosisClient()

        # Step 1: Create dataset record + get upload instructions
        try:
            dataset = client.create_dataset(project_id, file_path.name, file_size, ext)
        except PlatformAPIError as e:
            raise CLIError(f"Failed to create dataset: {e}") from e

        upload = dataset.upload
        if upload is None:
            raise CLIError("Server did not return upload instructions.")

        # Step 2: Upload file to S3
        is_multipart = upload.method == "multipart"
        if is_multipart:
            parts_label = f", {upload.total_parts} parts" if upload.total_parts else ""
            print(
                f"Uploading {file_path.name} ({_format_size(file_size)}, multipart{parts_label})..."
            )
        else:
            print(f"Uploading {file_path.name} ({_format_size(file_size)})...")

        progress_ctx, progress_cb = make_progress_bar(file_size)
        ctx = progress_ctx or contextlib.nullcontext()

        if is_multipart:
            try:
                with ctx:
                    parts = asyncio.run(
                        upload_file_multipart(
                            file_path, upload, progress_callback=progress_cb
                        )
                    )
                client.complete_upload(
                    dataset.id,
                    upload.s3_key,
                    ext,
                    upload_id=upload.upload_id,
                    parts=parts,
                )
            except KeyboardInterrupt:
                self._abort_multipart(client, dataset.id, upload.upload_id)
                raise CLIError("Upload cancelled by user.") from None
            except RuntimeError as e:
                self._abort_multipart(client, dataset.id, upload.upload_id)
                raise CLIError(f"Upload failed: {e}") from e
            except PlatformAPIError as e:
                self._abort_multipart(client, dataset.id, upload.upload_id)
                raise CLIError(f"Failed to complete upload: {e}") from e
        else:
            try:
                with ctx:
                    upload_file_simple(file_path, upload, progress_callback=progress_cb)
                client.complete_upload(dataset.id, upload.s3_key, ext)
            except KeyboardInterrupt:
                print("\nUpload interrupted.")
                raise CLIError("Upload cancelled by user.") from None
            except RuntimeError as e:
                raise CLIError(f"Upload failed: {e}") from e
            except PlatformAPIError as e:
                raise CLIError(f"Failed to complete upload: {e}") from e

        print(f"Upload complete. Dataset ID: {dataset.id}")
        print("Processing will continue on the platform. Check status in the web UI.")

        return 0

    def _run_list(self, args: argparse.Namespace) -> int:
        _require_auth()
        from osmosis_ai.platform.api.client import OsmosisClient

        project_id = _resolve_project_id(args)
        client = OsmosisClient()
        try:
            result = client.list_datasets(project_id)
        except PlatformAPIError as e:
            raise CLIError(str(e)) from e

        if not result.datasets:
            print("No datasets found.")
            return 0

        print(f"Datasets ({result.total_count}):")
        for d in result.datasets:
            status_info = f"[{d.status}]"
            if d.processing_step:
                status_info = f"[{d.status}: {d.processing_step}]"
            print(
                f"  {d.id[:8]}  {d.file_name}  {_format_size(d.file_size)}  {status_info}"
            )

        if result.has_more:
            print(f"  ... and {result.total_count - len(result.datasets)} more")

        return 0

    def _run_status(self, args: argparse.Namespace) -> int:
        _require_auth()
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
        try:
            ds = client.get_dataset(args.id)
        except PlatformAPIError as e:
            raise CLIError(str(e)) from e

        print(f"Dataset: {ds.file_name}")
        print(f"  ID:     {ds.id}")
        print(f"  Size:   {_format_size(ds.file_size)}")
        print(f"  Status: {ds.status}")
        if ds.processing_step:
            pct = (
                f" ({ds.processing_percent:.0f}%)"
                if ds.processing_percent is not None
                else ""
            )
            print(f"  Step:   {ds.processing_step}{pct}")
        if ds.error:
            print(f"  Error:  {ds.error}")
        return 0

    def _run_preview(self, args: argparse.Namespace) -> int:
        _require_auth()
        from osmosis_ai.platform.api.client import OsmosisClient

        client = OsmosisClient()
        try:
            ds = client.get_dataset(args.id)
        except PlatformAPIError as e:
            raise CLIError(str(e)) from e

        if ds.data_preview is None:
            if ds.status == "uploaded":
                print("No preview available for this dataset.")
            else:
                print(f"Dataset is still processing (status: {ds.status}).")
            return 0

        rows = ds.data_preview
        if isinstance(rows, list):
            limit = min(args.rows, len(rows))
            for row in rows[:limit]:
                print(row)
        else:
            print(rows)

        return 0

    def _run_delete(self, args: argparse.Namespace) -> int:
        _require_auth()
        from osmosis_ai.platform.api.client import OsmosisClient

        if not args.yes:
            confirm = input(
                f"Delete dataset '{args.id}'? This cannot be undone. [y/N] "
            )
            if confirm.lower() not in ("y", "yes"):
                print("Cancelled.")
                return 0

        client = OsmosisClient()
        try:
            client.delete_dataset(args.id)
        except PlatformAPIError as e:
            raise CLIError(str(e)) from e

        print("Dataset deleted.")
        return 0

    def _run_validate(self, args: argparse.Namespace) -> int:
        file_path = Path(args.file).resolve()
        if not file_path.exists():
            raise CLIError(f"File not found: {file_path}")

        ext = file_path.suffix.lstrip(".").lower()
        if ext not in VALID_EXTENSIONS:
            raise CLIError(
                f"Unsupported file type '.{ext}'. "
                f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise CLIError("File is empty.")
        if file_size > MAX_FILE_SIZE:
            raise CLIError(
                f"File too large ({_format_size(file_size)}). Maximum: {_format_size(MAX_FILE_SIZE)}"
            )

        # Format validation
        errors = _validate_file(file_path, ext)

        if errors:
            print("Validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

        print(f"Valid {ext} file: {file_path.name} ({_format_size(file_size)})")
        return 0


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

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []  # can't decode tail — skip tail validation

    lines = text.split("\n")
    # If we didn't read from the start, the first "line" may be partial
    if read_size < file_size:
        lines = lines[1:]
    # Drop trailing empty string produced by a final newline
    if lines and not lines[-1]:
        lines = lines[:-1]
    return lines[-n:]


def _validate_parquet(file_path: Path) -> list[str]:
    """Validate parquet file structure and required columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []  # pyarrow not installed, skip content validation

    errors = []
    try:
        pf = pq.ParquetFile(file_path)
        if len(pf.schema) == 0:
            errors.append("Parquet file has no columns")
        elif pf.metadata.num_rows == 0:
            errors.append("Parquet file has no rows")
        else:
            errors.extend(_check_required_columns(pf.schema_arrow.names))
    except Exception as e:
        errors.append(f"Invalid parquet file: {e}")
    return errors


def _validate_jsonl(file_path: Path) -> list[str]:
    """Validate JSONL: required columns + first/last 100 lines."""
    import json

    errors = []
    columns_checked = False
    file_fully_read = False

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 100:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if not columns_checked and isinstance(obj, dict):
                    errors.extend(_check_required_columns(obj.keys()))
                    columns_checked = True
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    return errors
        else:
            # Loop completed without break — entire file was read
            file_fully_read = True

    if file_fully_read or len(errors) >= 5:
        return errors

    # Validate last 100 lines
    tail_lines = _read_tail_lines(file_path, 100)
    for line in tail_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if not columns_checked and isinstance(obj, dict):
                errors.extend(_check_required_columns(obj.keys()))
                columns_checked = True
        except json.JSONDecodeError as e:
            errors.append(f"Near end of file: invalid JSON - {e}")
            if len(errors) >= 5:
                errors.append("... (showing first 5 errors)")
                break

    return errors


def _validate_csv(file_path: Path) -> list[str]:
    """Validate CSV: required columns + first/last 100 rows."""
    import csv

    errors = []
    num_cols = 0
    file_fully_read = False

    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return ["File has no header row"]

            errors.extend(_check_required_columns(header))
            num_cols = len(header)

            for i, row in enumerate(reader, 2):
                if i > 101:
                    break
                if len(row) != num_cols:
                    errors.append(
                        f"Row {i}: expected {num_cols} columns, got {len(row)}"
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        return errors
            else:
                file_fully_read = True
    except UnicodeDecodeError as e:
        errors.append(f"File encoding error: {e}")
        return errors

    if file_fully_read or len(errors) >= 5:
        return errors

    # Validate last 100 rows
    tail_lines = _read_tail_lines(file_path, 100)
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
    except csv.Error as e:
        errors.append(f"Near end of file: CSV parse error - {e}")

    return errors
