"""Handler for `osmosis dataset` commands."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth import PlatformAPIError

from .project import _require_auth, _resolve_project

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

CONTENT_TYPE_MAP = {
    "csv": "text/csv",
    "jsonl": "application/x-ndjson",
    "parquet": "application/vnd.apache.parquet",
}


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
        upload_parser.add_argument(
            "--no-wait",
            action="store_true",
            help="Don't wait for processing to complete",
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
        from osmosis_ai.platform.api.client import OsmosisClient
        from osmosis_ai.platform.api.upload import upload_file_to_presigned_url

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

        project_id = _resolve_project_id(args)
        client = OsmosisClient()

        # Step 1: Create dataset record
        print(f"Uploading {file_path.name} ({_format_size(file_size)})...")
        try:
            dataset = client.create_dataset(project_id, file_path.name, file_size)
        except PlatformAPIError as e:
            raise CLIError(f"Failed to create dataset: {e}") from e

        # Step 2: Get presigned upload URL
        try:
            upload_info = client.get_upload_url(dataset.id, ext)
        except PlatformAPIError as e:
            raise CLIError(f"Failed to get upload URL: {e}") from e

        # Step 3: Upload file
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")
        try:
            upload_file_to_presigned_url(
                file_path,
                upload_info.presigned_url,
                content_type,
                extra_headers=upload_info.upload_headers,
            )
        except RuntimeError as e:
            raise CLIError(f"Upload failed: {e}") from e

        # Step 4: Mark upload complete
        try:
            client.complete_upload(dataset.id, upload_info.s3_key, ext)
        except PlatformAPIError as e:
            raise CLIError(f"Failed to complete upload: {e}") from e

        print(f"Upload complete. Dataset ID: {dataset.id}")

        # Step 5: Optionally poll for processing
        if not args.no_wait:
            print("Waiting for processing...")
            ds = _poll_dataset_status(client, dataset.id)
            if ds.status == "uploaded":
                print("Processing complete.")
            elif ds.status == "error":
                print(f"Processing failed: {ds.error}")
                return 1
            else:
                print(f"Processing status: {ds.status}")

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
            pct = f" ({ds.processing_percent:.0f}%)" if ds.processing_percent else ""
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

        # Basic format validation
        errors = []
        if ext == "jsonl":
            errors = _validate_jsonl(file_path)
        elif ext == "csv":
            errors = _validate_csv(file_path)
        # Parquet validation requires pyarrow, skip for now

        if errors:
            print("Validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

        print(f"Valid {ext} file: {file_path.name} ({_format_size(file_size)})")
        return 0


def _poll_dataset_status(client, dataset_id: str, *, timeout: int = 600):
    """Poll dataset status until terminal state or timeout."""
    ds = client.get_dataset(dataset_id)
    if ds.is_terminal:
        return ds

    start = time.monotonic()
    interval = 2

    while time.monotonic() - start < timeout:
        time.sleep(interval)
        ds = client.get_dataset(dataset_id)
        if ds.is_terminal:
            return ds
        if time.monotonic() - start > 30:
            interval = 5

    return ds


def _validate_jsonl(file_path: Path) -> list[str]:
    """Basic JSONL validation - check first 100 lines parse as JSON."""
    import json

    errors = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > 100:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                if len(errors) >= 5:
                    errors.append("... (showing first 5 errors)")
                    break
    return errors


def _validate_csv(file_path: Path) -> list[str]:
    """Basic CSV validation - check it parses and has consistent columns."""
    import csv

    errors = []
    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return ["File has no header row"]
            num_cols = len(header)
            for i, row in enumerate(reader, 2):
                if i > 100:
                    break
                if len(row) != num_cols:
                    errors.append(
                        f"Row {i}: expected {num_cols} columns, got {len(row)}"
                    )
                    if len(errors) >= 5:
                        errors.append("... (showing first 5 errors)")
                        break
    except UnicodeDecodeError as e:
        errors.append(f"File encoding error: {e}")
    return errors
