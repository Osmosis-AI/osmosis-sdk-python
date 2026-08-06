from __future__ import annotations

import json
from dataclasses import asdict
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
benchmark = "DeepSWE"

[[agents]]
harness = "cursor-cli"
harness_api_key_secret = "CURSOR_API_KEY"

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


def _write_hosted_config(
    path: Path,
    *,
    benchmark: str,
    tasks: str = "",
) -> Path:
    path.write_text(
        f"""
[experiment]
benchmark = "{benchmark}"

{tasks}

[[agents]]
harness = "codex"

[agents.model]
type = "hosted"
base_model = "Qwen/Qwen3-8B"
lora_model_name = "benchmark-agent"

[execution]
judge_model = "openai/gpt-5"
judge_api_key_secret = "OPENAI_API_KEY"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _context(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_directory=workspace,
        git_identity=GIT_IDENTITY,
        repo_url=REPO_URL,
        credentials=FAKE_CREDENTIALS,
    )


class _FakeSubmitClient:
    def submit_benchmark_run(self, **kwargs: Any) -> SubmitBenchmarkRunResult:
        return SubmitBenchmarkRunResult(
            id="benchmark-run-1",
            name="bright-otter",
            status="pending",
            workflow_id="benchmark-run/benchmark-run-1",
            task_count=10,
            created_at="2026-07-25T00:00:00Z",
            platform_url="https://platform.osmosis.ai/acme/benchmarks/benchmark-run-1",
        )


def test_submit_sends_benchmark_config_and_returns_operation_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_config(workspace / "configs" / "benchmark" / "smoke.toml")
    context = _context(workspace)
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
        lambda *args, **kwargs: ({"OPENAI_API_KEY", "CURSOR_API_KEY"}, set()),
    )

    result = benchmark_module.submit(config_path, yes=True)

    assert captured["experiment_config"] == {"benchmark": "DeepSWE"}
    assert captured["tasks_config"] is None
    assert captured["execution_config"] == {"attempts_per_task": 2}
    assert captured["agents"][0]["model"]["model"] == "openai/gpt-5"
    assert captured["agents"][0]["harness_api_key_secret"] == "CURSOR_API_KEY"
    assert captured["credentials"] is FAKE_CREDENTIALS
    assert captured["git_identity"] == GIT_IDENTITY
    assert isinstance(result, OperationResult)
    assert result.operation == "benchmark.submit"
    assert result.resource is not None
    assert result.resource["task_count"] == 10
    assert result.resource["benchmark_name"] == "DeepSWE"
    assert result.resource["workflow_id"] == "benchmark-run/benchmark-run-1"
    assert result.resource["platform_url"] == (
        "https://platform.osmosis.ai/acme/benchmarks/benchmark-run-1"
    )
    assert "url" not in result.resource
    assert result.resource["config"]["agents"][0]["harness_api_key_secret"] == (
        "CURSOR_API_KEY"
    )


def test_submit_rejects_config_outside_benchmark_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_config(workspace / "smoke.toml")
    context = _context(workspace)
    monkeypatch.setattr(
        benchmark_module, "require_git_workspace_directory_context", lambda: context
    )

    with pytest.raises(CLIError, match="configs/benchmark"):
        benchmark_module.submit(config_path, yes=True)


@pytest.mark.parametrize("benchmark_name", ["HLE", " HLE ", " hLe "])
def test_submit_warns_before_confirmation_for_hle_without_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    benchmark_name: str,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_hosted_config(
        workspace / "configs" / "benchmark" / "hle.toml",
        benchmark=benchmark_name,
        tasks='[tasks]\ntask_names = ["hle__sample"]',
    )
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        lambda: _context(workspace),
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", _FakeSubmitClient)
    monkeypatch.setattr(
        benchmark_module,
        "_fetch_secret_scopes",
        lambda *args, **kwargs: ({"OPENAI_API_KEY", "HF_TOKEN"}, set()),
    )
    monkeypatch.setattr(
        benchmark_module.console,
        "print_warning",
        lambda message, **kwargs: events.append(
            ("warning", {"message": message, **kwargs})
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "require_confirmation",
        lambda *args, **kwargs: events.append(("confirmation", None)),
    )

    result = benchmark_module.submit(config_path, yes=True)

    assert result.status == "success"
    assert [event[0] for event in events] == ["warning", "confirmation"]
    warning = events[0][1]
    assert isinstance(warning, dict)
    assert warning["code"] == "HLE_PARITY_RECOMMENDED"
    assert 'task_set = "parity"' in str(warning["message"])


def test_submit_does_not_warn_when_hle_uses_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_hosted_config(
        workspace / "configs" / "benchmark" / "hle.toml",
        benchmark="HLE",
        tasks='[tasks]\ntask_set = "parity"',
    )
    warnings: list[str] = []

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        lambda: _context(workspace),
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", _FakeSubmitClient)
    monkeypatch.setattr(
        benchmark_module,
        "_fetch_secret_scopes",
        lambda *args, **kwargs: ({"OPENAI_API_KEY", "HF_TOKEN"}, set()),
    )
    monkeypatch.setattr(
        benchmark_module.console,
        "print_warning",
        lambda message, **kwargs: warnings.append(message),
    )

    result = benchmark_module.submit(config_path, yes=True)

    assert result.status == "success"
    assert warnings == []


def test_submit_warns_before_hle_missing_secret_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = _write_hosted_config(
        workspace / "configs" / "benchmark" / "hle.toml",
        benchmark="HLE",
    )
    warnings: list[dict[str, str]] = []

    monkeypatch.setattr(
        benchmark_module,
        "require_git_workspace_directory_context",
        lambda: _context(workspace),
    )
    monkeypatch.setattr(benchmark_module, "OsmosisClient", _FakeSubmitClient)
    monkeypatch.setattr(
        benchmark_module,
        "_fetch_secret_scopes",
        lambda *args, **kwargs: (set(), set()),
    )
    monkeypatch.setattr(
        benchmark_module.console,
        "print_warning",
        lambda message, **kwargs: warnings.append({"message": message, **kwargs}),
    )

    with pytest.raises(CLIError, match=r"OPENAI_API_KEY"):
        benchmark_module.submit(config_path, yes=True)

    assert warnings == [
        {
            "message": benchmark_module._HLE_PARITY_WARNING,
            "code": "HLE_PARITY_RECOMMENDED",
        }
    ]


def test_submit_sends_resolved_run_secrets_and_never_prints_a_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _make_workspace(tmp_path / "workspace")
    config_path = workspace / "configs" / "benchmark" / "secrets.toml"
    config_path.write_text(
        """
[experiment]
benchmark = "acme/custom@1.0"

[[agents]]
harness = "codex"

[agents.model]
type = "provider"
model = "openai/gpt-5"
api_key_secret = "OPENAI_API_KEY"

[secrets]
required = ["WEATHER_API_KEY"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    context = _context(workspace)
    captured: dict[str, Any] = {}

    class FakeClient:
        def submit_benchmark_run(self, **kwargs: Any) -> SubmitBenchmarkRunResult:
            captured.update(kwargs)
            return SubmitBenchmarkRunResult(
                id="benchmark-run-2",
                name="brave-otter",
                status="pending",
                workflow_id="benchmark-run/benchmark-run-2",
                task_count=3,
                created_at="2026-08-06T00:00:00Z",
                platform_url=None,
            )

    monkeypatch.setenv("WEATHER_API_KEY", "super-secret")
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

    assert captured["secrets"] == {
        "required": ["WEATHER_API_KEY"],
        "values": {"WEATHER_API_KEY": "super-secret"},
    }
    assert "super-secret" not in json.dumps(asdict(result), default=str)
