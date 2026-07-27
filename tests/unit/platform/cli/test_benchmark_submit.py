from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import osmosis_ai.platform.cli.benchmark as benchmark_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult
from osmosis_ai.platform.api.models import SubmitBenchmarkRunResult
from osmosis_ai.templates.catalog import required_workspace_paths

GIT_IDENTITY = "acme/workspace"
REPO_URL = "https://github.com/acme/workspace.git"
FAKE_CREDENTIALS = object()


def _make_workspace(root: Path) -> Path:
    for rel_path in required_workspace_paths():
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / "configs" / "benchmark").mkdir(parents=True)
    return root


def _write_config(path: Path) -> Path:
    path.write_text(
        """
[experiment]
benchmark = "Terminal-Bench 2.1"

[tasks]
task_set = "parity"

[[agents]]
harness = "codex"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[execution]
attempts_per_task = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_submit_sends_benchmark_config_and_returns_operation_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_config(workspace / "configs" / "benchmark" / "smoke.toml")
    context = SimpleNamespace(
        workspace_directory=workspace,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=FAKE_CREDENTIALS,
    )
    captured: dict[str, Any] = {}

    class FakeClient:
        def submit_benchmark_run(self, **kwargs: Any) -> SubmitBenchmarkRunResult:
            captured.update(kwargs)
            return SubmitBenchmarkRunResult(
                id="benchmark-run-1",
                name="bright-otter",
                status="pending",
                workflow_id="benchmark-run/benchmark-run-1",
                task_count=10,
                created_at="2026-07-25T00:00:00Z",
                platform_url="https://platform.osmosis.ai/acme/benchmarks/benchmark-run-1",
            )

    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", lambda: context
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", FakeClient)
    monkeypatch.setattr(
        benchmark_module,
        "_fetch_secret_scopes",
        lambda *args, **kwargs: ({"OPENAI_API_KEY"}, set()),
    )

    result = benchmark_module.submit(config_path, yes=True)

    assert captured["experiment_config"] == {"benchmark": "Terminal-Bench 2.1"}
    assert captured["tasks_config"] == {"task_set": "parity"}
    assert captured["execution_config"] == {"attempts_per_task": 2}
    assert captured["agents"][0]["model"]["model"] == "openai/gpt-5"
    assert captured["credentials"] is FAKE_CREDENTIALS
    assert captured["git_identity"] == GIT_IDENTITY
    assert isinstance(result, OperationResult)
    assert result.operation == "benchmark.submit"
    assert result.resource is not None
    assert result.resource["task_count"] == 10
    assert result.resource["benchmark_name"] == "Terminal-Bench 2.1"


def test_submit_rejects_config_outside_benchmark_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_config(workspace / "smoke.toml")
    context = SimpleNamespace(
        workspace_directory=workspace,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=FAKE_CREDENTIALS,
    )
    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", lambda: context
    )

    with pytest.raises(CLIError, match="configs/benchmark"):
        benchmark_module.submit(config_path, yes=True)
