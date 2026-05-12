"""Tests for dataset commands in JSON/plain output modes."""

from __future__ import annotations

import builtins
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.main as cli
import osmosis_ai.platform.api.client as api_client_module
import osmosis_ai.platform.api.download as download_module
import osmosis_ai.platform.api.upload as upload_module
import osmosis_ai.platform.cli.dataset as dataset_module
from osmosis_ai.platform.api.models import (
    DatasetDownloadInfo,
    DatasetFile,
    PaginatedDatasets,
    UploadInfo,
)

WORKSPACE_ID = "ws_1"
WORKSPACE_NAME = "ws-a"
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
    monkeypatch.setattr(
        dataset_module,
        "_require_subscription",
        lambda *, workspace_id, workspace_name: None,
    )
    return fake_credentials


def test_dataset_list_requires_linked_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "dataset", "list"])

    assert rc == 1
    assert "Not in an Osmosis project" in capsys.readouterr().err


def test_dataset_validate_is_project_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from osmosis_ai.cli.main import main

    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = main(["--json", "dataset", "validate", str(data)])

    assert rc == 0


def test_require_auth_no_workspace_reports_linked_project_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osmosis_ai.platform.cli.utils as utils_module
    from osmosis_ai.cli.errors import CLIError

    monkeypatch.setattr(
        utils_module,
        "require_credentials",
        lambda: pytest.fail("credentials should not be loaded"),
    )

    with pytest.raises(CLIError, match="linked Osmosis project"):
        utils_module._require_auth()


def _dataset(
    *,
    id: str = "ds_1",
    file_name: str = "train.jsonl",
    file_size: int = 100,
    status: str = "uploaded",
    data_preview=None,
) -> DatasetFile:
    return DatasetFile(
        id=id,
        file_name=file_name,
        file_size=file_size,
        status=status,
        data_preview=data_preview,
        created_at="2026-04-26T00:00:00Z",
        updated_at="2026-04-26T00:00:01Z",
    )


def test_dataset_list_json_envelope(monkeypatch, capsys) -> None:
    fake_credentials = _stub_workspace_context(monkeypatch)

    class FakeClient:
        def list_datasets(self, *, limit, offset, workspace_id, credentials=None):
            assert credentials is fake_credentials
            assert workspace_id == WORKSPACE_ID
            assert limit == 10
            assert offset == 0
            return PaginatedDatasets(
                datasets=[_dataset()],
                total_count=20,
                has_more=True,
                next_offset=10,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "dataset", "list", "--limit", "10"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["items"][0]["id"] == "ds_1"
    assert payload["total_count"] == 20
    assert payload["has_more"] is True
    assert payload["next_offset"] == 10
    assert payload["workspace"] == {"id": WORKSPACE_ID, "name": WORKSPACE_NAME}
    assert payload["project_root"] == str(PROJECT_ROOT)


def test_dataset_list_plain_emits_tab_separated_rows(monkeypatch, capsys) -> None:
    _stub_workspace_context(monkeypatch)

    class FakeClient:
        def list_datasets(self, *, limit, offset, workspace_id, credentials=None):
            assert workspace_id == WORKSPACE_ID
            return PaginatedDatasets(
                datasets=[
                    _dataset(id="ds_1", file_name="a.jsonl", status="uploaded"),
                    _dataset(id="ds_2", file_name="b.jsonl", status="pending"),
                ],
                total_count=2,
                has_more=False,
            )

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--plain", "dataset", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == [
        "a.jsonl\t[uploaded]\t100 B\t2026-04-26\tds_1",
        "b.jsonl\t[pending]\t100 B\t2026-04-26\tds_2",
    ]


def test_dataset_info_json_envelope(monkeypatch, capsys) -> None:
    _stub_workspace_context(monkeypatch)

    class FakeClient:
        def get_dataset(self, name, *, workspace_id, credentials=None):
            assert name == "ds_1"
            assert workspace_id == WORKSPACE_ID
            return _dataset(file_size=12345)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "dataset", "info", "ds_1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["data"]["id"] == "ds_1"
    assert payload["data"]["file_size"] == 12345
    assert payload["data"]["workspace"] == {"id": WORKSPACE_ID, "name": WORKSPACE_NAME}
    assert payload["data"]["project_root"] == str(PROJECT_ROOT)


def test_dataset_preview_json_includes_rows(monkeypatch, capsys) -> None:
    _stub_workspace_context(monkeypatch)
    rows = [{"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"}]

    class FakeClient:
        def get_dataset(self, name, *, workspace_id, credentials=None):
            assert workspace_id == WORKSPACE_ID
            return _dataset(data_preview=rows)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "dataset", "preview", "ds_1", "--rows", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["available"] is True
    assert payload["data"]["rows"] == rows
    assert payload["data"]["workspace"] == {"id": WORKSPACE_ID, "name": WORKSPACE_NAME}
    assert payload["data"]["project_root"] == str(PROJECT_ROOT)


def test_dataset_validate_json_envelope(tmp_path: Path, capsys) -> None:
    file_path = tmp_path / "train.jsonl"
    file_path.write_text(
        json.dumps({"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"})
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["--json", "dataset", "validate", str(file_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is True
    assert payload["data"]["file_name"] == "train.jsonl"


def test_dataset_validate_json_includes_parquet_warning_when_pyarrow_missing(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    real_import = builtins.__import__

    def _block_pyarrow(name: str, *args, **kwargs):
        if name == "pyarrow.parquet" or name == "pyarrow":
            raise ImportError("mocked missing pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pyarrow)
    file_path = tmp_path / "train.parquet"
    file_path.write_bytes(b"fake parquet data")

    exit_code = cli.main(["--json", "dataset", "validate", str(file_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["data"]["valid"] is True
    assert payload["data"]["warnings"]
    assert "pyarrow not installed" in payload["data"]["warnings"][0]


def test_dataset_upload_json_stdout_is_one_envelope(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    fake_credentials = _stub_workspace_context(monkeypatch)
    file_path = tmp_path / "train.jsonl"
    file_path.write_text(
        json.dumps({"system_prompt": "s", "user_prompt": "u", "ground_truth": "g"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        upload_module,
        "make_progress_bar",
        lambda _size, **_kwargs: (nullcontext(), lambda _done, _total: None),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_file_simple",
        lambda _file_path, _upload_info, progress_callback=None: None,
    )

    class FakeClient:
        def create_dataset(
            self, file_name, file_size, extension, *, workspace_id, credentials=None
        ):
            assert credentials is fake_credentials
            assert workspace_id == WORKSPACE_ID
            return DatasetFile(
                id="ds_1",
                file_name=file_name,
                file_size=file_size,
                status="created",
                upload=UploadInfo(
                    method="simple",
                    s3_key="uploads/train",
                    presigned_url="https://example.com/upload",
                ),
            )

        def complete_upload(
            self, file_id, parts=None, *, workspace_id, credentials=None
        ):
            assert file_id == "ds_1"
            assert credentials is fake_credentials
            assert workspace_id == WORKSPACE_ID
            return _dataset(id=file_id)

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)

    exit_code = cli.main(["--json", "dataset", "upload", str(file_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "dataset.upload"
    assert payload["resource"]["id"] == "ds_1"
    assert payload["resource"]["workspace"] == {
        "id": WORKSPACE_ID,
        "name": WORKSPACE_NAME,
    }
    assert payload["resource"]["project_root"] == str(PROJECT_ROOT)
    assert "Uploading" not in captured.out


def test_dataset_download_json_includes_output_path(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    _stub_workspace_context(monkeypatch)
    output_path = tmp_path / "out.jsonl"

    class FakeClient:
        def get_dataset(self, name, *, workspace_id, credentials=None):
            assert workspace_id == WORKSPACE_ID
            return _dataset()

        def get_dataset_download_url(self, name, *, workspace_id, credentials=None):
            assert workspace_id == WORKSPACE_ID
            return DatasetDownloadInfo(
                presigned_url="https://example.com/train.jsonl",
                expires_in=3600,
            )

    def fake_download_file(
        url,
        *,
        output,
        default_filename,
        expected_size,
        overwrite,
        output_is_directory,
    ):
        return output_path

    monkeypatch.setattr(api_client_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(download_module, "download_file", fake_download_file)

    exit_code = cli.main(
        [
            "--json",
            "dataset",
            "download",
            "ds_1",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["operation"] == "dataset.download"
    assert payload["resource"]["output_path"] == str(output_path)
    assert payload["resource"]["workspace"] == {
        "id": WORKSPACE_ID,
        "name": WORKSPACE_NAME,
    }
    assert payload["resource"]["project_root"] == str(PROJECT_ROOT)
