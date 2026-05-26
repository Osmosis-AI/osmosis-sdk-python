"""Tests for dataset upload core helpers."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.platform.api.models import DatasetFile, UploadInfo

GIT_IDENTITY = "acme/rollouts"


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
    def test_create_dataset_loading_message_says_uploading_dataset(self, monkeypatch):
        """Dataset creation uses the upload-oriented loading message."""
        import osmosis_ai.platform.cli.dataset as dataset_module

        messages: list[str] = []

        class FakeClient:
            def create_dataset(
                self, file_name, file_size, ext, *, git_identity, credentials=None
            ):
                assert file_name == "data"
                assert file_size == 2
                assert ext == "jsonl"
                assert git_identity == GIT_IDENTITY
                return _make_fake_dataset()

        def fake_platform_call(message, call, *, output_console=None):
            messages.append(message)
            return call()

        monkeypatch.setattr(dataset_module, "platform_call", fake_platform_call)

        result = dataset_module._create_dataset_for_upload(
            client=FakeClient(),
            dataset_name="data",
            file_size=2,
            ext="jsonl",
            overwrite=False,
            git_identity=GIT_IDENTITY,
            credentials=None,
        )

        assert result.id == "dataset-1"
        assert messages == ["Uploading dataset..."]

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
                self, file_name, file_size, ext, *, git_identity, credentials=None
            ):
                calls["create"] = True
                assert file_name == "data"
                assert ext == "jsonl"
                assert git_identity == GIT_IDENTITY
                assert credentials is fake_credentials
                return fake_dataset

            def complete_upload(
                self, file_id, parts=None, *, file_extension=None, git_identity, credentials=None
            ):
                calls["complete"] = True
                assert file_id == "dataset-1"
                assert parts is None
                assert git_identity == GIT_IDENTITY
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
            git_identity=GIT_IDENTITY,
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
                self, file_name, file_size, ext, *, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
                created["file_name"] = file_name
                return _make_fake_dataset(file_name=file_name, file_size=file_size)

            def complete_upload(
                self, file_id, parts=None, *, file_extension=None, git_identity, credentials=None
            ):
                assert git_identity == GIT_IDENTITY
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
            git_identity=GIT_IDENTITY,
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
            def create_dataset(self, *args, git_identity, **kwargs):
                assert git_identity == GIT_IDENTITY
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
                git_identity=GIT_IDENTITY,
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
            def create_dataset(self, *args, git_identity, **kwargs):
                assert git_identity == GIT_IDENTITY
                return _make_fake_dataset(file_size=file_size)

            def abort_upload(self, file_id, *, git_identity, credentials=None):
                assert git_identity == GIT_IDENTITY
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
                git_identity=GIT_IDENTITY,
                credentials=None,
            )

        assert aborted.get("called")

    def test_duplicate_name_without_overwrite_raises_guided_conflict(
        self, monkeypatch, tmp_path
    ):
        """_perform_upload points users to --overwrite on duplicate names."""
        import osmosis_ai.platform.api.client as api_client_module
        from osmosis_ai.cli.errors import CLIError
        from osmosis_ai.platform.auth import PlatformAPIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        conflict = PlatformAPIError(
            "A dataset with this name already exists",
            status_code=409,
            details={
                "error": "A dataset with this name already exists",
                "existing_dataset_id": "dataset-old",
            },
        )

        class FakeClient:
            def create_dataset(self, *args, git_identity, **kwargs):
                assert git_identity == GIT_IDENTITY
                raise conflict

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        from osmosis_ai.platform.cli.dataset import _perform_upload

        with pytest.raises(CLIError) as exc_info:
            _perform_upload(
                file_path=file_path,
                ext="jsonl",
                file_size=2,
                git_identity=GIT_IDENTITY,
                credentials=None,
            )

        assert exc_info.value.code == "CONFLICT"
        assert exc_info.value.details["existing_dataset_id"] == "dataset-old"
        assert "--overwrite" in exc_info.value.message
        assert "osmosis dataset upload" not in exc_info.value.message
        assert str(file_path) not in exc_info.value.message

    def test_overwrite_retries_create_with_existing_dataset_id(
        self, monkeypatch, tmp_path
    ):
        """_perform_upload retries create with overwrite_dataset_id before S3 upload."""
        import osmosis_ai.platform.api.client as api_client_module
        import osmosis_ai.platform.api.upload as upload_module
        from osmosis_ai.platform.auth import PlatformAPIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        file_size = file_path.stat().st_size
        calls: list[dict] = []

        class FakeClient:
            def create_dataset(
                self,
                file_name,
                file_size,
                ext,
                *,
                git_identity,
                credentials=None,
                overwrite_dataset_id=None,
            ):
                calls.append(
                    {
                        "file_name": file_name,
                        "overwrite_dataset_id": overwrite_dataset_id,
                    }
                )
                assert git_identity == GIT_IDENTITY
                if overwrite_dataset_id is None:
                    raise PlatformAPIError(
                        "A dataset with this name already exists",
                        status_code=409,
                        details={
                            "error": "A dataset with this name already exists",
                            "existing_dataset_id": "dataset-old",
                        },
                    )
                assert overwrite_dataset_id == "dataset-old"
                return _make_fake_dataset(file_name=file_name, file_size=file_size)

            def complete_upload(
                self, file_id, parts=None, *, file_extension=None, git_identity, credentials=None
            ):
                assert file_id == "dataset-1"
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
            lambda _fp, _info, progress_callback=None: None,
        )

        from osmosis_ai.platform.cli.dataset import _perform_upload

        result = _perform_upload(
            file_path=file_path,
            ext="jsonl",
            file_size=file_size,
            git_identity=GIT_IDENTITY,
            credentials=None,
            overwrite=True,
        )

        assert result.status == "uploaded"
        assert calls == [
            {"file_name": "data", "overwrite_dataset_id": None},
            {"file_name": "data", "overwrite_dataset_id": "dataset-old"},
        ]

    def test_conflict_without_existing_dataset_id_is_not_treated_as_overwriteable(
        self, monkeypatch, tmp_path
    ):
        """Guard conflicts from the platform should surface unchanged."""
        import osmosis_ai.platform.api.client as api_client_module
        from osmosis_ai.platform.auth import PlatformAPIError

        file_path = tmp_path / "data.jsonl"
        file_path.write_text("{}")
        conflict = PlatformAPIError(
            "This dataset is still processing.",
            status_code=409,
            details={"error": "This dataset is still processing."},
        )

        class FakeClient:
            def create_dataset(self, *args, git_identity, **kwargs):
                assert git_identity == GIT_IDENTITY
                raise conflict

        monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

        from osmosis_ai.platform.cli.dataset import _perform_upload

        with pytest.raises(PlatformAPIError) as exc_info:
            _perform_upload(
                file_path=file_path,
                ext="jsonl",
                file_size=2,
                git_identity=GIT_IDENTITY,
                credentials=None,
                overwrite=True,
            )

        assert exc_info.value is conflict


class TestUploadCommand:
    def test_yes_skips_interactive_confirmation(self, monkeypatch, tmp_path):
        import osmosis_ai.platform.cli.dataset as dataset_module

        file_path = tmp_path / "data.jsonl"
        row = '{"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"}\n'
        file_path.write_text(row * 4)
        fake_context = type(
            "Context",
            (),
            {
                "credentials": object(),
                "git_identity": GIT_IDENTITY,
                "workspace_directory": tmp_path,
                "repo_url": "https://github.com/acme/rollouts.git",
            },
        )()
        uploaded = _make_fake_dataset(
            file_name="data", file_size=file_path.stat().st_size
        )

        monkeypatch.setattr(
            dataset_module,
            "require_git_workspace_directory_context",
            lambda: fake_context,
        )
        monkeypatch.setattr(
            dataset_module,
            "_perform_upload",
            lambda **_kwargs: uploaded,
        )
        monkeypatch.setattr(
            dataset_module,
            "confirm",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("confirmation prompt should be skipped")
            ),
        )

        with override_output_context(format=OutputFormat.rich, interactive=True):
            result = dataset_module.upload(str(file_path), yes=True)

        assert result.operation == "dataset.upload"
        assert result.status == "success"
