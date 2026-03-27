"""Regression tests for dataset commands using explicit workspace context."""

from __future__ import annotations

import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.api.upload as upload_module
import osmosis_ai.platform.cli.dataset as dataset_module
from osmosis_ai.platform.api.models import DatasetFile, PaginatedDatasets, UploadInfo


def test_list_datasets_uses_active_workspace(monkeypatch) -> None:
    calls: dict[str, object | None] = {}
    fake_credentials = object()

    monkeypatch.setattr(
        dataset_module, "_require_auth", lambda: ("ws-b", fake_credentials)
    )

    class FakeClient:
        def list_datasets(
            self,
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
            assert file_name == "data.jsonl"
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
                    s3_key="uploads/data.jsonl",
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
                file_name="data.jsonl",
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
