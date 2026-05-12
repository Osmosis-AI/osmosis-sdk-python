"""Regression tests for dataset commands using explicit Git context."""

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
import osmosis_ai.platform.cli.utils as utils_module
from osmosis_ai.cli.console import Console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.models import (
    DatasetDownloadInfo,
    DatasetFile,
    PaginatedDatasets,
    UploadInfo,
)

GIT_IDENTITY = "acme/rollouts"
REPO_URL = "https://github.com/acme/rollouts.git"
PROJECT_ROOT = Path("/repo")


def assert_git_context(data: dict[str, object]) -> None:
    assert data["project_root"] == "/repo"
    assert data["git"] == {
        "identity": GIT_IDENTITY,
        "remote_url": REPO_URL,
    }
    assert "workspace" not in data


def _stub_git_context(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_credentials = object()
    context = SimpleNamespace(
        project_root=PROJECT_ROOT,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=fake_credentials,
    )
    monkeypatch.setattr(
        dataset_module,
        "require_git_project_context",
        lambda: context,
    )
    return fake_credentials


def test_list_datasets_uses_git_context(monkeypatch) -> None:
    calls: dict[str, object | None] = {}
    fake_credentials = _stub_git_context(monkeypatch)

    class FakeClient:
        def list_datasets(
            self,
            limit: int = 50,
            offset: int = 0,
            *,
            credentials=None,
            git_identity: str,
        ) -> PaginatedDatasets:
            calls["credentials"] = credentials
            calls["git_identity"] = git_identity
            return PaginatedDatasets(datasets=[], total_count=0, has_more=False)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    result = dataset_module.list_datasets()

    assert calls == {
        "credentials": fake_credentials,
        "git_identity": GIT_IDENTITY,
    }
    assert result.total_count == 0
    assert_git_context(result.extra)


def test_upload_passes_git_context_to_api_calls_without_subscription_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_git_context(monkeypatch)
    output = StringIO()

    file_path = tmp_path / "data.jsonl"
    file_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        utils_module,
        "load_subscription_status",
        lambda *_args, **_kwargs: pytest.fail("subscription cache should not be read"),
    )
    monkeypatch.setattr(dataset_module, "_validate_file", lambda _path, _ext: [])
    monkeypatch.setattr(
        dataset_module, "console", Console(file=output, force_terminal=False)
    )
    monkeypatch.setattr(
        dataset_module,
        "platform_entity_url",
        lambda identity, entity, item_id: (
            f"https://example.com/{identity}/{entity}/{item_id}"
        ),
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
            git_identity: str,
        ) -> DatasetFile:
            assert file_name == "data"
            assert file_size > 0
            assert extension == "jsonl"
            calls["create_credentials"] = credentials
            calls["create_git_identity"] = git_identity
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
            git_identity: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            assert parts is None
            calls["complete_credentials"] = credentials
            calls["complete_git_identity"] = git_identity
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=2,
                status="uploaded",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    result = dataset_module.upload(str(file_path))

    assert calls == {
        "create_credentials": fake_credentials,
        "create_git_identity": GIT_IDENTITY,
        "complete_credentials": fake_credentials,
        "complete_git_identity": GIT_IDENTITY,
    }
    assert result.operation == "dataset.upload"
    assert result.status == "success"
    assert result.resource["id"] == "dataset-1"
    assert result.resource["status"] == "uploaded"
    assert_git_context(result.resource)
    assert (
        f"https://example.com/{GIT_IDENTITY}/datasets/dataset-1"
        in result.display_next_steps[0]
    )


def test_download_uses_git_context_and_saves_file(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_git_context(monkeypatch)
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
            git_identity: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            calls["get_credentials"] = credentials
            calls["get_git_identity"] = git_identity
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
            git_identity: str,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            calls["download_url_credentials"] = credentials
            calls["download_url_git_identity"] = git_identity
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
        "get_git_identity": GIT_IDENTITY,
        "download_url_credentials": fake_credentials,
        "download_url_git_identity": GIT_IDENTITY,
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
    assert_git_context(result.resource)


def test_download_preserves_trailing_separator_as_directory_intent(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_git_context(monkeypatch)

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
            git_identity: str,
        ) -> DatasetFile:
            assert file_id == "dataset-1"
            assert credentials is fake_credentials
            calls["get_git_identity"] = git_identity
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
            git_identity: str,
        ) -> DatasetDownloadInfo:
            assert file_id == "dataset-1"
            assert credentials is fake_credentials
            calls["download_url_git_identity"] = git_identity
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
        "get_git_identity": GIT_IDENTITY,
        "download_url_git_identity": GIT_IDENTITY,
        "output": missing_dir,
        "output_is_directory": True,
    }


def test_download_rejects_processing_dataset(monkeypatch) -> None:
    calls: dict[str, object] = {}
    fake_credentials = _stub_git_context(monkeypatch)

    class FakeClient:
        def get_dataset(
            self,
            file_id: str,
            *,
            credentials=None,
            git_identity: str,
        ) -> DatasetFile:
            assert credentials is fake_credentials
            calls["git_identity"] = git_identity
            return DatasetFile(
                id=file_id,
                file_name="data",
                file_size=4,
                status="processing",
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    with pytest.raises(CLIError, match="still processing"):
        dataset_module.download("dataset-1")

    assert calls == {"git_identity": GIT_IDENTITY}
