"""Regression tests for dataset commands using explicit workspace context."""

from __future__ import annotations

import os
from io import StringIO

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.api.download as download_module
import osmosis_ai.platform.api.upload as upload_module
import osmosis_ai.platform.cli.dataset as dataset_module
from osmosis_ai.cli.console import Console
from osmosis_ai.errors import CLIError
from osmosis_ai.platform.api.models import (
    DatasetDownloadInfo,
    DatasetFile,
    PaginatedDatasets,
    UploadInfo,
)


def test_list_datasets_uses_active_workspace(monkeypatch) -> None:
    calls: dict[str, object | None] = {}
    fake_credentials = object()

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    class FakeClient:
        def list_datasets(
            self,
            limit: int = 50,
            offset: int = 0,
            *,
            credentials=None,
        ) -> PaginatedDatasets:
            calls["credentials"] = credentials
            return PaginatedDatasets(datasets=[], total_count=0, has_more=False)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    dataset_module.list_datasets()

    assert calls == {
        "credentials": fake_credentials,
    }


def test_upload_passes_active_workspace_context_to_subscription_and_api_calls(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = object()
    output = StringIO()

    file_path = tmp_path / "data.jsonl"
    file_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    def fake_require_subscription(*, workspace_name: str) -> None:
        calls["workspace_name"] = workspace_name

    monkeypatch.setattr(
        dataset_module, "_require_subscription", fake_require_subscription
    )
    monkeypatch.setattr(dataset_module, "_validate_file", lambda _path, _ext: [])
    monkeypatch.setattr(dataset_module, "is_interactive", lambda: False)
    monkeypatch.setattr(
        dataset_module, "console", Console(file=output, force_terminal=False)
    )
    monkeypatch.setattr(
        dataset_module,
        "platform_entity_url",
        lambda ws, entity, item_id: f"https://example.com/{ws}/{entity}/{item_id}",
    )
    from contextlib import nullcontext

    monkeypatch.setattr(
        upload_module,
        "make_progress_bar",
        lambda _size: (nullcontext(), lambda _done, _total: None),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_file_simple",
        lambda _file_path, _upload_info, progress_callback=None: None,
    )

    class FakeClient:
        def create_dataset(
            self,
            file_name: str,
            file_size: int,
            extension: str,
            *,
            credentials=None,
        ) -> DatasetFile:
            assert file_name == "data"
            assert file_size > 0
            assert extension == "jsonl"
            calls["create_credentials"] = credentials
            return DatasetFile(
                id="dataset-1",
                file_name=file_name,
                file_size=file_size,
                status="created",
                upload=UploadInfo(
                    method="simple",
                    s3_key="uploads/data",
                    presigned_url="https://example.com/upload",
                ),
            )

        def complete_upload(
            self,
            file_id: str,
            parts: list[dict] | None = None,
            *,
            credentials=None,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            assert parts is None
            calls["complete_credentials"] = credentials
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=2,
                status="uploaded",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    dataset_module.upload(str(file_path))

    assert calls == {
        "workspace_name": "ws-b",
        "create_credentials": fake_credentials,
        "complete_credentials": fake_credentials,
    }
    rendered = output.getvalue()
    assert "Dataset uploaded: data" in rendered
    assert "https://example.com/ws-b/datasets/dataset-1" in rendered


def test_download_uses_active_workspace_context_and_saves_file(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = object()
    output = StringIO()

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )
    monkeypatch.setattr(
        dataset_module,
        "console",
        Console(file=output, force_terminal=False),
    )

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            calls["get_credentials"] = credentials
            return DatasetFile(
                id="dataset-1",
                file_name="data",
                file_size=4,
                status="uploaded",
            )

        def get_dataset_download_url(
            self,
            file_id: str,
            *,
            credentials=None,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            calls["download_url_credentials"] = credentials
            return DatasetDownloadInfo(
                presigned_url="https://example.com/data.jsonl",
                expires_in=3600,
            )

    def fake_download_file(
        url: str,
        *,
        output,
        default_filename: str,
        expected_size: int,
        overwrite: bool,
        output_is_directory: bool,
    ):
        calls["download_url"] = url
        calls["output"] = output
        calls["default_filename"] = default_filename
        calls["expected_size"] = expected_size
        calls["overwrite"] = overwrite
        calls["output_is_directory"] = output_is_directory
        return tmp_path / "data.jsonl"

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(download_module, "download_file", fake_download_file)

    dataset_module.download("dataset-1", output=str(tmp_path), overwrite=True)

    assert calls == {
        "get_credentials": fake_credentials,
        "download_url_credentials": fake_credentials,
        "download_url": "https://example.com/data.jsonl",
        "output": tmp_path,
        "default_filename": "data.jsonl",
        "expected_size": 4,
        "overwrite": True,
        "output_is_directory": False,
    }
    assert "Dataset downloaded:" in output.getvalue()


def test_download_preserves_trailing_separator_as_directory_intent(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = object()

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            return DatasetFile(
                id="dataset-1",
                file_name="data",
                file_size=4,
                status="uploaded",
            )

        def get_dataset_download_url(
            self,
            file_id: str,
            *,
            credentials=None,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            return DatasetDownloadInfo(
                presigned_url="https://example.com/data.jsonl",
                expires_in=3600,
            )

    def fake_download_file(
        url: str,
        *,
        output,
        default_filename: str,
        expected_size: int,
        overwrite: bool,
        output_is_directory: bool,
    ):
        calls["output"] = output
        calls["output_is_directory"] = output_is_directory
        return tmp_path / "data.jsonl"

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(download_module, "download_file", fake_download_file)

    missing_dir = tmp_path / "missing"
    dataset_module.download("dataset-1", output=f"{missing_dir}{os.sep}")

    assert calls == {
        "output": missing_dir,
        "output_is_directory": True,
    }


def test_download_rejects_processing_dataset(monkeypatch) -> None:
    fake_credentials = object()

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
        ) -> DatasetFile:
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=4,
                status="processing",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    with pytest.raises(CLIError, match="still processing"):
        dataset_module.download("dataset-1")
