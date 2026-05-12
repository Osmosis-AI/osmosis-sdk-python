"""Tests for dataset upload core helpers."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from osmosis_ai.platform.api.models import DatasetFile, UploadInfo

WORKSPACE_ID = "ws_1"


def _make_fake_dataset(
    file_name: str = "data",
    file_size: int = 2,
    method: str = "simple",
) -> DatasetFile:
    return DatasetFile(
        id="dataset-1",
        file_name=file_name,
        file_size=file_size,
        status="created",
        upload=UploadInfo(
            method=method,
            s3_key=f"uploads/{file_name}",
            presigned_url="https://example.com/upload" if method == "simple" else None,
            upload_id="up-1" if method == "multipart" else None,
            part_size=5 * 1024 * 1024 if method == "multipart" else None,
            total_parts=1 if method == "multipart" else None,
            presigned_urls=(
                [{"part_number": 1, "presigned_url": "https://example.com/part1"}]
                if method == "multipart"
                else None
            ),
        ),
    )


class TestPerformUpload:
    def test_simple_upload_flow(self, monkeypatch, tmp_path):
        """_perform_upload creates dataset, uploads to S3, and completes."""
        import osmosis_ai.platform.api.client as api_client_module
        import osmosis_ai.platform.api.upload as upload_module

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        file_size = file_path.stat().st_size

        calls: dict[str, bool] = {}
        fake_credentials = object()
        fake_dataset = _make_fake_dataset(file_size=file_size)

        class FakeClient:
            def create_dataset(
                self, file_name, file_size, ext, *, workspace_id, credentials=None
            ):
                calls["create"] = True
                assert file_name == "data"
                assert ext == "jsonl"
                assert workspace_id == WORKSPACE_ID
                assert credentials is fake_credentials
                return fake_dataset

            def complete_upload(
                self, file_id, parts=None, *, workspace_id, credentials=None
            ):
                calls["complete"] = True
                assert file_id == "dataset-1"
                assert parts is None
                assert workspace_id == WORKSPACE_ID
                assert credentials is fake_credentials
                return DatasetFile(
                    id=file_id,
                    file_name="data",
                    file_size=file_size,
                    status="uploaded",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(
            upload_module,
            "make_progress_bar",
            lambda _size: (nullcontext(), lambda _done, _total: None),
        )
        monkeypatch.setattr(
            upload_module,
            "upload_file_simple",
            lambda _fp, _info, progress_callback=None: calls.update(
                {"s3_upload": True}
            ),
        )

        from osmosis_ai.platform.cli.dataset import _perform_upload

        result = _perform_upload(
            file_path=file_path,
            ext="jsonl",
            file_size=file_size,
            workspace_id=WORKSPACE_ID,
            credentials=fake_credentials,
        )

        assert result.id == "dataset-1"
        assert calls["create"]
        assert calls["s3_upload"]
        assert calls["complete"]

    def test_strips_extension_from_dataset_name(self, monkeypatch, tmp_path):
        """_perform_upload sends the dataset name without the file extension."""
        import osmosis_ai.platform.api.client as api_client_module
        import osmosis_ai.platform.api.upload as upload_module

        file_path = tmp_path / "agent.eval.v2.jsonl"
        file_path.write_text("{}")
        file_size = file_path.stat().st_size

        created: dict[str, str] = {}

        class FakeClient:
            def create_dataset(
                self, file_name, file_size, ext, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                created["file_name"] = file_name
                return _make_fake_dataset(file_name=file_name, file_size=file_size)

            def complete_upload(
                self, file_id, parts=None, *, workspace_id, credentials=None
            ):
                assert workspace_id == WORKSPACE_ID
                return DatasetFile(
                    id=file_id,
                    file_name="agent.eval.v2",
                    file_size=file_size,
                    status="uploaded",
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(
            upload_module,
            "make_progress_bar",
            lambda _size: (nullcontext(), lambda _done, _total: None),
        )
        monkeypatch.setattr(
            upload_module,
            "upload_file_simple",
            lambda _fp, _info, progress_callback=None: None,
        )

        from osmosis_ai.platform.cli.dataset import _perform_upload

        _perform_upload(
            file_path=file_path,
            ext="jsonl",
            file_size=file_size,
            workspace_id=WORKSPACE_ID,
            credentials=None,
        )

        assert created == {"file_name": "agent.eval.v2"}

    def test_raises_on_missing_upload_info(self, monkeypatch, tmp_path):
        """_perform_upload raises CLIError if server returns no upload info."""
        import osmosis_ai.platform.api.client as api_client_module
        from osmosis_ai.cli.errors import CLIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")

        class FakeClient:
            def create_dataset(self, *args, workspace_id, **kwargs):
                assert workspace_id == WORKSPACE_ID
                return DatasetFile(
                    id="dataset-1",
                    file_name="data",
                    file_size=2,
                    status="created",
                    upload=None,
                )

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        from osmosis_ai.platform.cli.dataset import _perform_upload

        with pytest.raises(CLIError, match="upload instructions"):
            _perform_upload(
                file_path=file_path,
                ext="jsonl",
                file_size=2,
                workspace_id=WORKSPACE_ID,
                credentials=None,
            )

    def test_abort_on_keyboard_interrupt(self, monkeypatch, tmp_path):
        """_perform_upload aborts and raises CLIError on KeyboardInterrupt."""
        import osmosis_ai.platform.api.client as api_client_module
        import osmosis_ai.platform.api.upload as upload_module
        from osmosis_ai.cli.errors import CLIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        file_size = file_path.stat().st_size

        aborted = {}

        class FakeClient:
            def create_dataset(self, *args, workspace_id, **kwargs):
                assert workspace_id == WORKSPACE_ID
                return _make_fake_dataset(file_size=file_size)

            def abort_upload(self, file_id, *, workspace_id, credentials=None):
                assert workspace_id == WORKSPACE_ID
                aborted["called"] = True

            def complete_upload(self, *args, **kwargs):
                pass

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
        monkeypatch.setattr(
            upload_module,
            "make_progress_bar",
            lambda _size: (nullcontext(), lambda _done, _total: None),
        )
        monkeypatch.setattr(
            upload_module,
            "upload_file_simple",
            lambda _fp, _info, progress_callback=None: (_ for _ in ()).throw(
                KeyboardInterrupt
            ),
        )

        from osmosis_ai.platform.cli.dataset import _perform_upload

        with pytest.raises(CLIError, match="cancelled"):
            _perform_upload(
                file_path=file_path,
                ext="jsonl",
                file_size=file_size,
                workspace_id=WORKSPACE_ID,
                credentials=None,
            )

        assert aborted.get("called")
