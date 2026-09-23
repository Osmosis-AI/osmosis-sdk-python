"""Tests for rollout backends constructed from serve TOML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.rollout import serve as serve_module


def test_load_simple_config(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        """\
backend = "simple"

[simple]
workflow = "demo.workflow:Workflow"
grader = "demo.grader:Grader"
workflow_config = "demo.workflow:workflow_config"
grader_config = "demo.grader:grader_config"
"""
    )

    config, rollout_dir = serve_module.load_serve_config(config_path)

    assert config.backend == "simple"
    assert config.simple.workflow == "demo.workflow:Workflow"
    assert config.simple.grader == "demo.grader:Grader"
    assert rollout_dir == tmp_path


def test_load_harbor_config(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        """\
backend = "harbor"

[harbor]
tasks_dir = "dataset"
task_mode = "dataset"
agent = "mini-swe-agent"
environment = "daytona"
concurrency = 12
cleanup_successful_trials = false
patch_dockerfile_with_sdk = false
agent_setup_timeout_sec = 90
max_queue_depth = 24

[harbor.native_agent_kwargs]
max_seq_len = 8192

[harbor.environment_kwargs]
auto_stop_interval_mins = 30
"""
    )

    config, rollout_dir = serve_module.load_serve_config(config_path)

    assert config.backend == "harbor"
    assert config.harbor.tasks_dir == Path("dataset")
    assert config.harbor.native_agent_kwargs == {"max_seq_len": 8192}
    assert config.harbor.environment_kwargs == {"auto_stop_interval_mins": 30}
    assert rollout_dir == tmp_path


@pytest.mark.parametrize(
    "content,match",
    [
        ('backend = "unknown"\n', "backend"),
        ('backend = "simple"\n', "simple"),
        (
            'backend = "simple"\n[simple]\nworkflow = "demo:Workflow"\nextra = 1\n',
            "extra",
        ),
        (
            'backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\nconcurrency = 0\n',
            "concurrency",
        ),
    ],
)
def test_load_config_rejects_invalid_schema(
    tmp_path: Path, content: str, match: str
) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(content)

    with pytest.raises(CLIError, match=match):
        serve_module.load_serve_config(config_path)


def test_server_port_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PORT", "8123")
    assert serve_module._server_port(None) == 8123
    assert serve_module._server_port(9000) == 9000


@pytest.mark.parametrize("value", ["invalid", "0", "65536"])
def test_server_port_rejects_invalid_environment(monkeypatch, value: str) -> None:
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PORT", value)
    with pytest.raises(CLIError, match="_OSMOSIS_ROLLOUT_PORT"):
        serve_module._server_port(None)


def test_run_server_builds_app_and_starts_uvicorn(monkeypatch) -> None:
    import uvicorn

    import osmosis_ai.rollout.server as server_package

    backend = object()
    app = object()
    captured: dict[str, Any] = {}

    def fake_create_rollout_server(*, backend: Any) -> object:
        captured["backend"] = backend
        return app

    def fake_uvicorn_run(server_app: object, *, host: str, port: int) -> None:
        captured["app"] = server_app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(
        server_package, "create_rollout_server", fake_create_rollout_server
    )
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setenv("_OSMOSIS_ROLLOUT_PORT", "8123")

    serve_module._run_server(backend=backend, host="127.0.0.1", port=None)

    assert captured == {
        "backend": backend,
        "app": app,
        "host": "127.0.0.1",
        "port": 8123,
    }


def test_serve_simple_constructs_backend_in_config_directory(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        """\
backend = "simple"

[simple]
workflow = "demo.workflow:Workflow"
grader = "demo.grader:Grader"
workflow_config = "demo.workflow:workflow_config"
grader_config = "demo.grader:grader_config"
"""
    )
    original_cwd = Path.cwd()
    original_sys_path = sys.path.copy()
    captured: dict[str, Any] = {}

    class FakeLocalBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured["backend_kwargs"] = kwargs
            captured["backend_cwd"] = Path.cwd()
            captured["import_root"] = sys.path[0]

    def fake_run_server(*, backend: Any, host: str, port: int | None) -> None:
        captured["backend"] = backend
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(
        "osmosis_ai.rollout.backend.local.LocalBackend", FakeLocalBackend
    )
    monkeypatch.setattr(serve_module, "_run_server", fake_run_server)

    serve_module.serve(config_path, host="127.0.0.1", port=8123)

    assert captured["backend_kwargs"] == {
        "workflow": "demo.workflow:Workflow",
        "grader": "demo.grader:Grader",
        "workflow_config": "demo.workflow:workflow_config",
        "grader_config": "demo.grader:grader_config",
    }
    assert captured["backend_cwd"] == tmp_path
    assert captured["import_root"] == str(tmp_path)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert Path.cwd() == original_cwd
    assert sys.path == original_sys_path


def test_serve_harbor_constructs_backend_from_config(
    monkeypatch, tmp_path: Path
) -> None:
    tasks_dir = tmp_path / "dataset"
    tasks_dir.mkdir()
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        """\
backend = "harbor"

[harbor]
tasks_dir = "dataset"
task_mode = "dataset"
agent = "mini-swe-agent"
environment = "daytona"
concurrency = 12
native_model_name = "openai/test-model"
trials_dir = "trials"
cleanup_successful_trials = false
patch_dockerfile_with_sdk = false
agent_setup_timeout_sec = 90
max_queue_depth = 24

[harbor.native_agent_kwargs]
max_seq_len = 8192

[harbor.environment_kwargs]
auto_stop_interval_mins = 30
"""
    )
    captured: dict[str, Any] = {}

    class FakeTrialQueue:
        def __init__(self, *, n_concurrent: int) -> None:
            self.n_concurrent = n_concurrent

    class FakeHarborBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured["backend_kwargs"] = kwargs

    def fake_run_server(*, backend: Any, host: str, port: int | None) -> None:
        captured["backend"] = backend
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("harbor.trial.queue.TrialQueue", FakeTrialQueue)
    monkeypatch.setattr(
        "osmosis_ai.rollout.backend.harbor.HarborBackend", FakeHarborBackend
    )
    monkeypatch.setattr(serve_module, "_run_server", fake_run_server)

    serve_module.serve(config_path, host="127.0.0.1", port=8124)

    backend_kwargs = captured["backend_kwargs"]
    assert backend_kwargs["tasks_dir"] == tasks_dir.resolve()
    assert backend_kwargs["task_mode"] == "dataset"
    assert backend_kwargs["agent"] == "mini-swe-agent"
    assert backend_kwargs["model_name"] == "openai/test-model"
    assert backend_kwargs["native_agent_kwargs"] == {"max_seq_len": 8192}
    assert backend_kwargs["code_dir"] == tmp_path.resolve()
    assert backend_kwargs["environment_config"].type.value == "daytona"
    assert backend_kwargs["environment_config"].kwargs == {
        "auto_stop_interval_mins": 30
    }
    assert backend_kwargs["orchestrator"].n_concurrent == 12
    assert backend_kwargs["cleanup_successful_trials"] is False
    assert backend_kwargs["patch_dockerfile_with_sdk"] is False
    assert backend_kwargs["max_queue_depth"] == 24
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8124


@pytest.mark.parametrize("field", ["code_dir", "bundle"])
def test_harbor_rejects_internal_packaging_fields(tmp_path: Path, field: str) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        f'backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\n{field} = "."\n'
    )

    with pytest.raises(CLIError, match=field):
        serve_module.load_serve_config(config_path)
