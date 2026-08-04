from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.main as cli
import osmosis_ai.platform.cli.benchmark as benchmark_module
import osmosis_ai.platform.cli.run_download as run_download_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.models import (
    BenchmarkRunDetail,
    RunDownloadFile,
    RunDownloadManifest,
    RunDownloadURL,
    RunDownloadURLBatch,
)
from osmosis_ai.platform.auth import PlatformAPIError

GIT_IDENTITY = "acme/workspace"

FILES = {
    "summary": [RunDownloadFile("summary.csv", 10, token="export-1")],
    "results": [RunDownloadFile("results.csv", 20, token="export-1")],
    "artifacts": [
        RunDownloadFile(
            "artifacts/result_01/logs/agent.log",
            30,
            token="result-01",
        )
    ],
    "logs": [RunDownloadFile("logs.txt", 40)],
}


def _detail(status: str = "finished") -> BenchmarkRunDetail:
    return BenchmarkRunDetail(
        id="benchmark-run-1",
        name="hle-smoke",
        status=status,
        benchmark_name="HLE",
        created_at="2026-07-30T00:00:00Z",
    )


def _stub_context(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> object:
    credentials = object()
    context = SimpleNamespace(
        workspace_directory=workspace,
        git_identity=GIT_IDENTITY,
        repo_url="https://github.com/acme/workspace.git",
        credentials=credentials,
    )
    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        lambda: context,
    )
    return credentials


def _fake_client(
    *,
    status: str = "finished",
    manifest_error: Exception | None = None,
    manifest_files: list[RunDownloadFile] | None = None,
):
    manifest_calls: list[tuple[str, ...]] = []
    url_calls: list[list[RunDownloadFile]] = []

    class FakeClient:
        def get_benchmark_run(self, name_or_id, *, git_identity, credentials=None):
            assert name_or_id == "hle-smoke"
            return _detail(status)

        def get_benchmark_run_download_manifest(
            self, run_id, *, types, git_identity, credentials=None
        ):
            if manifest_error is not None:
                raise manifest_error
            assert run_id == "benchmark-run-1"
            manifest_calls.append(tuple(types))
            files = (
                manifest_files
                if manifest_files is not None
                else [item for kind in types for item in FILES[kind]]
            )
            return RunDownloadManifest(
                files=files,
                totals={"files": len(files), "bytes": sum(item.size for item in files)},
            )

        def get_benchmark_run_download_urls(
            self, run_id, *, items, git_identity, credentials=None
        ):
            assert run_id == "benchmark-run-1"
            url_calls.append(list(items))
            return RunDownloadURLBatch(
                items=[
                    RunDownloadURL(
                        path=item.path,
                        token=item.token,
                        url=f"https://example.test/{item.path}",
                    )
                    for item in items
                ]
            )

    FakeClient.manifest_calls = manifest_calls
    FakeClient.url_calls = url_calls
    return FakeClient


def _stub_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url, destination, *, expected_size=None):
        size = expected_size or 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * size)
        return size

    monkeypatch.setattr(run_download_module, "download_file_to", fake_download)


def test_default_benchmark_download_uses_run_scoped_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_context(monkeypatch, tmp_path)
    fake_client = _fake_client()
    monkeypatch.setattr(benchmark_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    exit_code = cli.main(["--json", "benchmark", "runs", "download", "hle-smoke"])
    captured = capsys.readouterr()

    assert exit_code == 0
    envelope = json.loads(captured.out)
    resource = envelope["resource"]
    root = tmp_path / ".osmosis" / "benchmarks" / "hle-smoke"
    assert envelope["operation"] == "benchmark.download"
    assert resource["benchmark_run"] == {
        "id": "benchmark-run-1",
        "name": "hle-smoke",
    }
    assert resource["selected_types"] == ["summary", "results"]
    assert fake_client.manifest_calls == [("summary", "results")]
    assert (root / "summary.csv").is_file()
    assert (root / "results.csv").is_file()
    assert not (root / "logs.txt").exists()


def test_all_benchmark_download_accepts_stable_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    fake_client = _fake_client()
    monkeypatch.setattr(benchmark_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    result = benchmark_module.download(
        "hle-smoke",
        output=None,
        types="all",
        overwrite=False,
        yes=True,
    )

    root = tmp_path / ".osmosis" / "benchmarks" / "hle-smoke"
    assert result.status == "success"
    assert (root / "artifacts" / "result_01" / "logs" / "agent.log").is_file()
    assert (root / "logs.txt").is_file()


@pytest.mark.parametrize("status", ["pending", "queued"])
def test_benchmark_download_rejects_pending_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    fake_client = _fake_client(status=status)
    monkeypatch.setattr(benchmark_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    exit_code = cli.main(["--json", "benchmark", "runs", "download", "hle-smoke"])
    captured = capsys.readouterr()

    assert exit_code == 1
    envelope = json.loads(captured.err)
    assert envelope["command"] == "benchmark runs download"
    assert envelope["error"]["code"] == "CONFLICT"
    assert envelope["error"]["message"] == (
        "Outputs are not yet available for pending or queued benchmark runs."
    )
    assert fake_client.manifest_calls == []


def test_benchmark_download_allows_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    fake_client = _fake_client(status="running")
    monkeypatch.setattr(benchmark_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    result = benchmark_module.download(
        "hle-smoke",
        output=None,
        types="summary",
        overwrite=False,
        yes=True,
    )

    assert result.status == "success"
    assert fake_client.manifest_calls == [("summary",)]


def test_benchmark_download_rejects_unknown_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    monkeypatch.setattr(benchmark_module, "OsmosisClient", _fake_client())

    exit_code = cli.main(
        ["--json", "benchmark", "runs", "download", "hle-smoke", "--type", "metrics"]
    )

    assert exit_code == 1


def test_benchmark_download_translates_missing_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        benchmark_module,
        "OsmosisClient",
        _fake_client(manifest_error=PlatformAPIError("not found", status_code=404)),
    )

    with pytest.raises(CLIError, match="platform may not support benchmark downloads"):
        benchmark_module.download(
            "hle-smoke",
            output=None,
            types="summary",
            overwrite=False,
            yes=True,
        )


def test_benchmark_download_rejects_paths_outside_fixed_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        benchmark_module,
        "OsmosisClient",
        _fake_client(
            manifest_files=[
                RunDownloadFile("../summary.csv", 10),
                RunDownloadFile("summary.json", 10),
            ]
        ),
    )

    with pytest.raises(CLIError, match="no usable run-scoped paths"):
        benchmark_module.download(
            "hle-smoke",
            output=None,
            types="summary",
            overwrite=False,
            yes=True,
        )
