"""Tests for the run-output download CLI contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import osmosis_ai.cli.main as cli
import osmosis_ai.platform.cli.eval as eval_module
import osmosis_ai.platform.cli.run_download as run_download_module
from osmosis_ai.platform.api.download import DownloadHTTPError
from osmosis_ai.platform.api.models import (
    EvalRunMetrics,
    EvalRunMetricsOverview,
    EvaluationRunDetail,
    RunDownloadFile,
    RunDownloadManifest,
    RunDownloadURL,
    RunDownloadURLBatch,
)
from osmosis_ai.platform.auth import PlatformAPIError

GIT_IDENTITY = "acme/rollouts"

FILES = {
    "metrics": [RunDownloadFile("metrics.json", 10)],
    "trajectories": [
        RunDownloadFile("summary.jsonl", 20),
        RunDownloadFile("trajectories/row_3_run_0.json", 30, rollout_id="rollout-3-0"),
    ],
    "artifacts": [
        RunDownloadFile(
            "artifacts/row_3_run_0/logs/agent.log",
            40,
            rollout_id="rollout-3-0",
        )
    ],
    "logs": [RunDownloadFile("logs.txt", 50)],
}


def _stub_git_context(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> object:
    credentials = object()
    context = SimpleNamespace(
        workspace_directory=workspace,
        git_identity=GIT_IDENTITY,
        repo_url="https://github.com/acme/rollouts.git",
        credentials=credentials,
    )
    monkeypatch.setattr(
        eval_module,
        "require_git_workspace_directory_context",
        lambda: context,
    )
    return credentials


def _detail(status: str = "finished") -> EvaluationRunDetail:
    return EvaluationRunDetail(
        id="er_12345678",
        name="my-run",
        status=status,
        created_at="2026-07-01T00:00:00Z",
    )


def _make_fake_client(
    *,
    status: str = "finished",
    manifest_error: Exception | None = None,
    manifest_files: list[RunDownloadFile] | None = None,
):
    manifest_calls: list[dict[str, object]] = []
    url_calls: list[list[RunDownloadFile]] = []

    class FakeClient:
        def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
            assert name_or_id == "my-run"
            assert git_identity == GIT_IDENTITY
            return _detail(status)

        def get_eval_run_download_manifest(
            self,
            run_id,
            *,
            types,
            rows=None,
            git_identity,
            credentials=None,
        ):
            if manifest_error is not None:
                raise manifest_error
            assert run_id == "er_12345678"
            manifest_calls.append({"types": tuple(types), "rows": rows})
            files = (
                manifest_files
                if manifest_files is not None
                else [item for kind in types for item in FILES[kind]]
            )
            return RunDownloadManifest(
                files=files,
                totals={"files": len(files), "bytes": sum(f.size for f in files)},
            )

        def get_eval_run_download_urls(
            self,
            run_id,
            *,
            items,
            git_identity,
            credentials=None,
        ):
            assert run_id == "er_12345678"
            url_calls.append(list(items))
            return RunDownloadURLBatch(
                items=[
                    RunDownloadURL(
                        path=item.path,
                        rollout_id=item.rollout_id,
                        url=f"https://example.com/{item.path}?size={item.size}",
                    )
                    for item in items
                ],
                expires_in=900,
            )

    FakeClient.manifest_calls = manifest_calls
    FakeClient.url_calls = url_calls
    return FakeClient


def _stub_download(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    downloaded: list[str] = []

    def fake_download_file_to(
        url: str,
        destination: Path,
        *,
        expected_size: int | None = None,
    ) -> int:
        size = expected_size if expected_size is not None else 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * size)
        downloaded.append(url)
        return size

    monkeypatch.setattr(run_download_module, "download_file_to", fake_download_file_to)
    return downloaded


def test_default_download_is_metrics_and_trajectories(monkeypatch, tmp_path, capsys):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    exit_code = cli.main(["--json", "eval", "download", "my-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    resource = json.loads(captured.out)["resource"]
    expected = tmp_path / ".osmosis" / "evals" / "my-run"
    assert resource["selected_types"] == ["metrics", "trajectories"]
    assert resource["output_path"] == str(expected)
    assert fake_client.manifest_calls == [
        {"types": ("metrics", "trajectories"), "rows": None}
    ]
    assert (expected / "metrics.json").is_file()
    assert (expected / "summary.jsonl").is_file()
    assert (expected / "trajectories" / "row_3_run_0.json").is_file()
    assert not (expected / "artifacts").exists()


def test_eval_info_uses_same_run_scoped_metrics_path(monkeypatch, tmp_path):
    _stub_git_context(monkeypatch, tmp_path)
    server_metrics = b'{"summary":{"pass_rate":1}}\n'

    class InfoClient:
        def get_eval_run(self, name_or_id, *, git_identity, credentials=None):
            return _detail()

        def get_eval_run_metrics(self, eval_run_id, *, git_identity, credentials=None):
            return EvalRunMetrics(
                eval_run_id=eval_run_id,
                status="finished",
                overview=EvalRunMetricsOverview(
                    duration_ms=None,
                    total_samples=1,
                    completed_samples=1,
                    graded=1,
                    passed=1,
                    failed=0,
                    skipped=0,
                    pass_rate=1.0,
                    pass_threshold=0.5,
                    tokens_used=10,
                ),
                reward_stats=None,
                pass_at_k=[],
            )

        def get_eval_run_download_manifest(
            self, eval_run_id, *, types, git_identity, credentials=None
        ):
            assert types == ["metrics"]
            return RunDownloadManifest(
                files=[
                    RunDownloadFile(
                        "metrics.json",
                        len(server_metrics),
                        rollout_id="export-token",
                    )
                ],
                totals={"files": 1, "bytes": len(server_metrics)},
            )

        def get_eval_run_download_urls(
            self, eval_run_id, *, items, git_identity, credentials=None
        ):
            return RunDownloadURLBatch(
                items=[
                    RunDownloadURL(
                        path=items[0].path,
                        rollout_id=items[0].rollout_id,
                        url="https://example.com/metrics.json",
                    )
                ],
                expires_in=900,
            )

    monkeypatch.setattr(eval_module, "OsmosisClient", InfoClient)

    def download_server_metrics(url, destination, *, expected_size=None):
        destination.write_bytes(server_metrics)
        return len(server_metrics)

    monkeypatch.setattr(
        run_download_module, "download_file_to", download_server_metrics
    )

    result = eval_module.info("my-run", output=None)

    expected = tmp_path / ".osmosis/evals/my-run/metrics.json"
    assert result.data["output_path"] == str(expected)
    assert expected.read_bytes() == server_metrics


def test_type_selector_replaces_default_and_rows_are_forwarded(
    monkeypatch, tmp_path, capsys
):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)
    output = tmp_path / "custom-root"

    exit_code = cli.main(
        [
            "--json",
            "eval",
            "download",
            "my-run",
            "--type",
            "artifacts,logs",
            "--rows",
            "3,7,10-20",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    resource = json.loads(captured.out)["resource"]
    assert resource["selected_types"] == ["artifacts", "logs"]
    assert resource["rows"] == "3,7,10-20"
    assert fake_client.manifest_calls == [
        {"types": ("artifacts", "logs"), "rows": "3,7,10-20"}
    ]
    assert (output / "artifacts/row_3_run_0/logs/agent.log").is_file()
    assert (output / "logs.txt").is_file()
    assert not (output / "metrics.json").exists()


def test_type_all_expands_all_eval_categories(monkeypatch, tmp_path, capsys):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    exit_code = cli.main(["--json", "eval", "download", "my-run", "--type", "all"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["resource"]["selected_types"] == [
        "metrics",
        "trajectories",
        "artifacts",
        "logs",
    ]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--type", "unknown"], "Unknown --type"),
        (["--type", "all,logs"], "cannot be combined"),
        (["--rows", "10-3"], "start exceeds end"),
        (["--rows", "3,,7"], "--rows must use syntax"),
    ],
)
def test_invalid_selectors_fail_before_manifest(
    monkeypatch, tmp_path, capsys, args, message
):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)

    exit_code = cli.main(["--json", "eval", "download", "my-run", *args])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert message in json.loads(captured.err)["error"]["message"]
    assert fake_client.manifest_calls == []


def test_resume_skips_matching_sizes_and_overwrite_forces_download(
    monkeypatch, tmp_path, capsys
):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    downloaded = _stub_download(monkeypatch)
    output = tmp_path / "run"
    existing = output / "metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x" * 10)

    assert (
        cli.main(
            [
                "--json",
                "eval",
                "download",
                "my-run",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)["resource"]
    assert first["files_skipped"] == 1
    assert len(downloaded) == 2

    assert (
        cli.main(
            [
                "--json",
                "eval",
                "download",
                "my-run",
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)["resource"]
    assert second["files_skipped"] == 0
    assert len(downloaded) == 5


def test_presigned_urls_are_requested_in_batches_of_at_most_500(
    monkeypatch, tmp_path, capsys
):
    files = [
        RunDownloadFile(f"trajectories/row_{row}_run_0.json", 1) for row in range(501)
    ]
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client(manifest_files=files)
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)

    exit_code = cli.main(
        ["--json", "eval", "download", "my-run", "--type", "trajectories"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["resource"]["files_downloaded"] == 501
    assert [len(batch) for batch in fake_client.url_calls] == [500, 1]


def test_large_download_requires_yes_in_json(monkeypatch, tmp_path, capsys):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    _stub_download(monkeypatch)
    monkeypatch.setattr(run_download_module, "DOWNLOAD_CONFIRM_THRESHOLD_BYTES", 1)

    exit_code = cli.main(["--json", "eval", "download", "my-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERACTIVE_REQUIRED"
    assert fake_client.url_calls == []

    assert cli.main(["--json", "eval", "download", "my-run", "--yes"]) == 0


def test_invalid_manifest_paths_are_reported_and_reserved_manifest_is_skipped(
    monkeypatch, tmp_path, capsys
):
    files = [
        RunDownloadFile("metrics.json", 10),
        RunDownloadFile("../evil.json", 10),
        RunDownloadFile("artifacts/row_3_run_0/manifest.json", 10),
    ]
    _stub_git_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        eval_module, "OsmosisClient", _make_fake_client(manifest_files=files)
    )
    _stub_download(monkeypatch)

    exit_code = cli.main(["--json", "eval", "download", "my-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "partial"
    assert payload["resource"]["files_downloaded"] == 1
    assert {item["path"] for item in payload["resource"]["files_failed"]} == {
        "../evil.json",
        "artifacts/row_3_run_0/manifest.json",
    }
    assert not (tmp_path / "evil.json").exists()


def test_nested_artifact_manifest_is_downloadable(monkeypatch, tmp_path, capsys):
    nested = "artifacts/row_3_run_0/logs/manifest.json"
    files = [RunDownloadFile(nested, 10, rollout_id="rollout-3-0")]
    _stub_git_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        eval_module, "OsmosisClient", _make_fake_client(manifest_files=files)
    )
    _stub_download(monkeypatch)

    exit_code = cli.main(
        ["--json", "eval", "download", "my-run", "--type", "artifacts"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["resource"]["files_downloaded"] == 1
    assert (tmp_path / ".osmosis/evals/my-run" / nested).is_file()


def test_manifest_cannot_cross_existing_symlink_outside_run_root(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "run"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "trajectories").symlink_to(outside, target_is_directory=True)
    files = [RunDownloadFile("trajectories/row_3_run_0.json", 10)]
    _stub_git_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        eval_module, "OsmosisClient", _make_fake_client(manifest_files=files)
    )

    exit_code = cli.main(
        [
            "--json",
            "eval",
            "download",
            "my-run",
            "--type",
            "trajectories",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "no usable run-scoped paths" in json.loads(captured.err)["error"]["message"]
    assert list(outside.iterdir()) == []


def test_per_file_retries_continue_and_report_only_permanent_failures(
    monkeypatch, tmp_path, capsys
):
    _stub_git_context(monkeypatch, tmp_path)
    monkeypatch.setattr(eval_module, "OsmosisClient", _make_fake_client())
    attempts: dict[str, int] = {}

    def flaky_download(url, destination, *, expected_size=None):
        attempts[url] = attempts.get(url, 0) + 1
        if "metrics.json" in url and attempts[url] < 3:
            raise RuntimeError("connection reset")
        if "summary.jsonl" in url:
            raise RuntimeError(f"still broken {url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * expected_size)
        return expected_size

    monkeypatch.setattr(run_download_module, "download_file_to", flaky_download)
    monkeypatch.setattr(run_download_module, "DOWNLOAD_RETRY_BASE_SECONDS", 0)

    exit_code = cli.main(["--json", "eval", "download", "my-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    resource = json.loads(captured.out)["resource"]
    assert resource["files_downloaded"] == 2
    assert [item["path"] for item in resource["files_failed"]] == ["summary.jsonl"]
    assert "example.com" not in resource["files_failed"][0]["error"]
    assert "<redacted URL>" in resource["files_failed"][0]["error"]
    metrics_url = next(url for url in attempts if "metrics.json" in url)
    assert attempts[metrics_url] == 3


def test_error_message_redacts_mixed_case_presigned_url() -> None:
    error = RuntimeError("failed HTTPS://example.com/file?token=secret")

    assert run_download_module._safe_error_message(error) == "failed <redacted URL>"


def test_expired_presigned_url_refreshes_by_typed_status(monkeypatch, tmp_path, capsys):
    _stub_git_context(monkeypatch, tmp_path)
    fake_client = _make_fake_client()
    monkeypatch.setattr(eval_module, "OsmosisClient", fake_client)
    attempts = 0

    def expired_once(url, destination, *, expected_size=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DownloadHTTPError(403, "expired")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * expected_size)
        return expected_size

    monkeypatch.setattr(run_download_module, "download_file_to", expired_once)
    monkeypatch.setattr(run_download_module, "DOWNLOAD_RETRY_BASE_SECONDS", 0)

    exit_code = cli.main(["--json", "eval", "download", "my-run", "--type", "metrics"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["resource"]["files_downloaded"] == 1
    assert attempts == 2
    assert [len(batch) for batch in fake_client.url_calls] == [1, 1]


def test_missing_manifest_route_has_upgrade_hint(monkeypatch, tmp_path, capsys):
    _stub_git_context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        eval_module,
        "OsmosisClient",
        _make_fake_client(manifest_error=PlatformAPIError("Not found", 404)),
    )

    exit_code = cli.main(["--json", "eval", "download", "my-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert (
        "download route was not found" in json.loads(captured.err)["error"]["message"]
    )
