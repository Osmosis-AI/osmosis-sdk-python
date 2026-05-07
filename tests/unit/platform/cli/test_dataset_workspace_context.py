"""Regression tests for dataset commands using explicit workspace context."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.api.download as download_module
import osmosis_ai.platform.api.upload as upload_module
import osmosis_ai.platform.cli.dataset as dataset_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.models import (
    DatasetDownloadInfo,
    DatasetFile,
    PaginatedDatasets,
    UploadInfo,
)

WORKSPACE_ID = "ws_b_id"
WORKSPACE_NAME = "ws-b"
PROJECT_ROOT = Path("/tmp/osmosis-project")


def _stub_workspace_context(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_credentials = object()
    workspace = SimpleNamespace(
        project_root=PROJECT_ROOT,
        workspace_id=WORKSPACE_ID,
        workspace_name=WORKSPACE_NAME,
        credentials=fake_credentials,
    )
    monkeypatch.setattr(
        dataset_module,
        "require_workspace_context",
        lambda: workspace,
    )
    return fake_credentials


def test_list_datasets_uses_linked_workspace_context(monkeypatch) -> None:
    calls: dict[str, object | None] = {}
    fake_credentials = _stub_workspace_context(monkeypatch)

    class FakeClient:
        def list_datasets(
            self,
            limit: int = 50,
            offset: int = 0,
            *,
            credentials=None,
            workspace_id: str,
        ) -> PaginatedDatasets:
            calls["credentials"] = credentials
            calls["workspace_id"] = workspace_id
            return PaginatedDatasets(datasets=[], total_count=0, has_more=False)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    result = dataset_module.list_datasets()

    assert calls == {
        "credentials": fake_credentials,
        "workspace_id": WORKSPACE_ID,
    }
    assert result.total_count == 0


def test_upload_passes_linked_workspace_context_to_subscription_and_api_calls(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_workspace_context(monkeypatch)
    output = StringIO()

    file_path = tmp_path / "data.jsonl"
    file_path.write_text("{}", encoding="utf-8")

    def fake_require_subscription(*, workspace_id: str, workspace_name: str) -> None:
        calls["subscription_workspace_id"] = workspace_id
        calls["workspace_name"] = workspace_name

    monkeypatch.setattr(
        dataset_module, "_require_subscription", fake_require_subscription
    )
    monkeypatch.setattr(dataset_module, "_validate_file", lambda _path, _ext: [])
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
            workspace_id: str,
        ) -> DatasetFile:
            assert file_name == "data"
            assert file_size > 0
            assert extension == "jsonl"
            calls["create_credentials"] = credentials
            calls["create_workspace_id"] = workspace_id
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
            workspace_id: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            assert parts is None
            calls["complete_credentials"] = credentials
            calls["complete_workspace_id"] = workspace_id
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=2,
                status="uploaded",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    result = dataset_module.upload(str(file_path))

    assert calls == {
        "subscription_workspace_id": WORKSPACE_ID,
        "workspace_name": WORKSPACE_NAME,
        "create_credentials": fake_credentials,
        "create_workspace_id": WORKSPACE_ID,
        "complete_credentials": fake_credentials,
        "complete_workspace_id": WORKSPACE_ID,
    }
    assert result.operation == "dataset.upload"
    assert result.status == "success"
    assert result.resource["id"] == "dataset-1"
    assert result.resource["status"] == "uploaded"
    assert (
        f"https://example.com/{WORKSPACE_NAME}/datasets/dataset-1"
        in result.display_next_steps[0]
    )


def test_download_uses_linked_workspace_context_and_saves_file(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_workspace_context(monkeypatch)
    output = StringIO()

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
            workspace_id: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            calls["get_credentials"] = credentials
            calls["get_workspace_id"] = workspace_id
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
            workspace_id: str,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            calls["download_url_credentials"] = credentials
            calls["download_url_workspace_id"] = workspace_id
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

    result = dataset_module.download("dataset-1", output=str(tmp_path), overwrite=True)

    assert calls == {
        "get_credentials": fake_credentials,
        "get_workspace_id": WORKSPACE_ID,
        "download_url_credentials": fake_credentials,
        "download_url_workspace_id": WORKSPACE_ID,
        "download_url": "https://example.com/data.jsonl",
        "output": tmp_path,
        "default_filename": "data.jsonl",
        "expected_size": 4,
        "overwrite": True,
        "output_is_directory": False,
    }
    assert result.operation == "dataset.download"
    assert result.status == "success"
    assert result.resource["output_path"] == str(tmp_path / "data.jsonl")


def test_download_preserves_trailing_separator_as_directory_intent(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_workspace_context(monkeypatch)

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
            workspace_id: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            assert credentials is fake_credentials
            calls["get_workspace_id"] = workspace_id
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
            workspace_id: str,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            assert credentials is fake_credentials
            calls["download_url_workspace_id"] = workspace_id
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
        "get_workspace_id": WORKSPACE_ID,
        "download_url_workspace_id": WORKSPACE_ID,
        "output": missing_dir,
        "output_is_directory": True,
    }


def test_download_rejects_processing_dataset(monkeypatch) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_workspace_context(monkeypatch)

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
            workspace_id: str,
        ) -> DatasetFile:
            assert credentials is fake_credentials
            calls["workspace_id"] = workspace_id
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=4,
                status="processing",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    with pytest.raises(CLIError, match="still processing"):
        dataset_module.download("dataset-1")

    assert calls == {"workspace_id": WORKSPACE_ID}
