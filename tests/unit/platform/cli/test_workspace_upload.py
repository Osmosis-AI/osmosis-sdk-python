"""Tests for dataset upload within the workspace interactive flow."""

from __future__ import annotations

from contextlib import nullcontext
from io import StringIO

import pytest

from osmosis_ai.cli.console import Console
from osmosis_ai.platform.api.models import DatasetFile, UploadInfo

# ---------------------------------------------------------------------------
# _clean_file_path — drag-and-drop path normalization
# ---------------------------------------------------------------------------


class TestCleanFilePath:
    def test_strips_whitespace(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert _clean_file_path("  /path/to/file.jsonl  ") == "/path/to/file.jsonl"

    def test_strips_single_quotes(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert _clean_file_path("'/path/to/file.jsonl'") == "/path/to/file.jsonl"

    def test_strips_double_quotes(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert _clean_file_path('"/path/to/file.jsonl"') == "/path/to/file.jsonl"

    def test_handles_escaped_spaces(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert _clean_file_path("/path/to/my\\ file.jsonl") == "/path/to/my file.jsonl"

    def test_combined_whitespace_and_quotes(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert (
            _clean_file_path("  '/path/to/my\\ file.jsonl'  ")
            == "/path/to/my file.jsonl"
        )

    def test_plain_path_unchanged(self):
        from osmosis_ai.platform.cli.workspace import _clean_file_path

        assert _clean_file_path("/path/to/file.jsonl") == "/path/to/file.jsonl"


# ---------------------------------------------------------------------------
# _perform_upload — extracted core upload logic
# ---------------------------------------------------------------------------


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
            def create_dataset(self, file_name, file_size, ext, *, credentials=None):
                calls["create"] = True
                assert file_name == "data"
                assert ext == "jsonl"
                assert credentials is fake_credentials
                return fake_dataset

            def complete_upload(self, file_id, parts=None, *, credentials=None):
                calls["complete"] = True
                assert file_id == "dataset-1"
                assert parts is None  # simple upload
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
            def create_dataset(self, file_name, file_size, ext, *, credentials=None):
                created["file_name"] = file_name
                return _make_fake_dataset(file_name=file_name, file_size=file_size)

            def complete_upload(self, file_id, parts=None, *, credentials=None):
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
            credentials=None,
        )

        assert created == {"file_name": "agent.eval.v2"}

    def test_raises_on_missing_upload_info(self, monkeypatch, tmp_path):
        """_perform_upload raises CLIError if server returns no upload info."""
        import osmosis_ai.platform.api.client as api_client_module
        from osmosis_ai.errors import CLIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")

        class FakeClient:
            def create_dataset(self, *args, **kwargs):
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
                credentials=None,
            )

    def test_abort_on_keyboard_interrupt(self, monkeypatch, tmp_path):
        """_perform_upload aborts and raises CLIError on KeyboardInterrupt."""
        import osmosis_ai.platform.api.client as api_client_module
        import osmosis_ai.platform.api.upload as upload_module
        from osmosis_ai.errors import CLIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        file_size = file_path.stat().st_size

        aborted = {}

        class FakeClient:
            def create_dataset(self, *args, **kwargs):
                return _make_fake_dataset(file_size=file_size)

            def abort_upload(self, file_id, *, credentials=None):
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
                credentials=None,
            )

        assert aborted.get("called")


# ---------------------------------------------------------------------------
# _upload_dataset_interactive — workspace upload flow
# ---------------------------------------------------------------------------


class TestUploadDatasetInteractive:
    def _setup_mocks(self, monkeypatch, tmp_path, *, file_content="{}"):
        """Set up common mocks for interactive upload tests."""
        import osmosis_ai.platform.cli.dataset as dataset_module
        import osmosis_ai.platform.cli.workspace as workspace_module

        file_path = tmp_path / "data.jsonl"
        file_path.write_text(file_content)

        # Mock validation to pass
        monkeypatch.setattr(dataset_module, "_validate_file", lambda _p, _e: [])

        return file_path, dataset_module, workspace_module

    def test_prompts_and_uploads(self, monkeypatch, tmp_path):
        """Interactive upload prompts for file, validates, confirms, and uploads."""
        import osmosis_ai.platform.cli.workspace as workspace_module

        file_path, dataset_module, _ = self._setup_mocks(monkeypatch, tmp_path)
        output = StringIO()

        # Mock text() to return file path (simulating drag & drop)
        monkeypatch.setattr(workspace_module, "text", lambda msg, **kw: str(file_path))
        # Mock confirm() to approve
        monkeypatch.setattr(workspace_module, "confirm", lambda msg, **kw: True)
        monkeypatch.setattr(
            workspace_module, "console", Console(file=output, force_terminal=False)
        )
        monkeypatch.setattr(
            workspace_module,
            "platform_entity_url",
            lambda ws, entity, item_id: f"https://example.com/{ws}/{entity}/{item_id}",
        )

        uploaded = {}

        def fake_perform_upload(*, file_path, ext, file_size, credentials):
            uploaded["called"] = True
            return DatasetFile(
                id="ds-1", file_name="data", file_size=2, status="uploaded"
            )

        monkeypatch.setattr(dataset_module, "_perform_upload", fake_perform_upload)

        from osmosis_ai.platform.cli.workspace import _upload_dataset_interactive

        result = _upload_dataset_interactive(
            ws_name="my-ws",
            credentials=object(),
        )

        assert result is True
        assert uploaded["called"]
        rendered = output.getvalue()
        assert "Upload complete. Dataset: data" in rendered
        assert "https://example.com/my-ws/datasets/ds-1" in rendered

    def test_returns_false_on_cancelled_input(self, monkeypatch, tmp_path):
        """Returns False when user cancels the file path prompt."""
        import osmosis_ai.platform.cli.workspace as workspace_module

        monkeypatch.setattr(workspace_module, "text", lambda msg, **kw: None)

        from osmosis_ai.platform.cli.workspace import _upload_dataset_interactive

        result = _upload_dataset_interactive(
            ws_name="my-ws",
            credentials=object(),
        )

        assert result is False

    def test_returns_false_on_invalid_file(self, monkeypatch, tmp_path):
        """Returns False when the file path is invalid."""
        import osmosis_ai.platform.cli.workspace as workspace_module

        monkeypatch.setattr(
            workspace_module, "text", lambda msg, **kw: "/nonexistent/file.jsonl"
        )

        from osmosis_ai.platform.cli.workspace import _upload_dataset_interactive

        result = _upload_dataset_interactive(
            ws_name="my-ws",
            credentials=object(),
        )

        assert result is False

    def test_returns_false_on_declined_confirm(self, monkeypatch, tmp_path):
        """Returns False when user declines upload confirmation."""
        import osmosis_ai.platform.cli.workspace as workspace_module

        file_path, _dataset_module, _ = self._setup_mocks(monkeypatch, tmp_path)

        monkeypatch.setattr(workspace_module, "text", lambda msg, **kw: str(file_path))
        monkeypatch.setattr(workspace_module, "confirm", lambda msg, **kw: False)

        from osmosis_ai.platform.cli.workspace import _upload_dataset_interactive

        result = _upload_dataset_interactive(
            ws_name="my-ws",
            credentials=object(),
        )

        assert result is False

    def test_returns_false_on_validation_errors(self, monkeypatch, tmp_path):
        """Returns False when file content validation fails."""
        import osmosis_ai.platform.cli.dataset as dataset_module
        import osmosis_ai.platform.cli.workspace as workspace_module

        file_path = tmp_path / "bad.jsonl"
        file_path.write_text("not json")

        monkeypatch.setattr(workspace_module, "text", lambda msg, **kw: str(file_path))
        monkeypatch.setattr(
            dataset_module, "_validate_file", lambda _p, _e: ["bad format"]
        )

        from osmosis_ai.platform.cli.workspace import _upload_dataset_interactive

        result = _upload_dataset_interactive(
            ws_name="my-ws",
            credentials=object(),
        )

        assert result is False
