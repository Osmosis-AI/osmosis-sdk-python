"""Tests for rollout backends constructed from serve TOML."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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
agent = "mini-swe-agent"
environment = "daytona"
image_repository = "us-west1-docker.pkg.dev/acme/repo/harbor"
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
    assert config.harbor.native_agent_kwargs == {"max_seq_len": 8192}
    assert config.harbor.environment_kwargs == {"auto_stop_interval_mins": 30}
    assert config.harbor.image_repository == (
        "us-west1-docker.pkg.dev/acme/repo/harbor"
    )
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
        (
            'backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\ntasks_dir = "tasks"\n',
            "tasks_dir",
        ),
        (
            'backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\ntask_mode = "dataset"\n',
            "task_mode",
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

    serve_module.serve(
        config_path,
        harbor_dataset=None,
        host="127.0.0.1",
        port=8123,
    )

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
agent = "mini-swe-agent"
environment = "daytona"
image_repository = "us-west1-docker.pkg.dev/acme/repo/harbor"
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

    async def fake_materialize_dataset(source: str) -> Path:
        captured["dataset_source"] = source
        return tasks_dir

    monkeypatch.setattr(
        serve_module, "_materialize_harbor_dataset", fake_materialize_dataset
    )
    monkeypatch.setattr(serve_module, "_run_server", fake_run_server)

    serve_module.serve(
        config_path,
        harbor_dataset="org/math@sha256:abc",
        host="127.0.0.1",
        port=8124,
    )

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
    assert backend_kwargs["image_repository"] == (
        "us-west1-docker.pkg.dev/acme/repo/harbor"
    )
    assert backend_kwargs["orchestrator"].n_concurrent == 12
    assert backend_kwargs["cleanup_successful_trials"] is False
    assert backend_kwargs["patch_dockerfile_with_sdk"] is False
    assert backend_kwargs["max_queue_depth"] == 24
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8124
    assert captured["dataset_source"] == "org/math@sha256:abc"


@pytest.mark.parametrize("field", ["code_dir", "bundle"])
def test_harbor_rejects_internal_packaging_fields(tmp_path: Path, field: str) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        f'backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\n{field} = "."\n'
    )

    with pytest.raises(CLIError, match=field):
        serve_module.load_serve_config(config_path)


def test_harbor_requires_dataset_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text('backend = "harbor"\n[harbor]\nagent = "mini-swe-agent"\n')

    with pytest.raises(CLIError, match="--harbor-dataset is required"):
        serve_module.serve(
            config_path,
            harbor_dataset=None,
            host="127.0.0.1",
            port=8000,
        )


def test_simple_rejects_dataset_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text('backend = "simple"\n[simple]\nworkflow = "demo:Workflow"\n')

    with pytest.raises(CLIError, match="only be used"):
        serve_module.serve(
            config_path,
            harbor_dataset="./tasks",
            host="127.0.0.1",
            port=8000,
        )


def test_dataset_source_supports_local_and_remote_forms(tmp_path: Path) -> None:
    local = tmp_path / "tasks"
    local.mkdir()

    local_config = serve_module._dataset_config_for_source(str(local))
    package_config = serve_module._dataset_config_for_source("acme/code@sha256:abc")
    repo_config = serve_module._dataset_config_for_source(
        "https://github.com/acme/tasks.git@abc123"
    )

    assert local_config.path == local
    assert package_config.name == "acme/code"
    assert package_config.ref == "sha256:abc"
    assert repo_config.repo == "https://github.com/acme/tasks.git@abc123"


async def test_remote_dataset_is_materialized_in_content_addressed_cache(
    monkeypatch, tmp_path: Path
) -> None:
    task_id = object()

    class FakeTaskConfig:
        def model_dump_json(self) -> str:
            return '{"name":"acme/task","ref":"sha256:def"}'

        def get_task_id(self) -> object:
            return task_id

    class FakeDatasetConfig:
        async def get_task_configs(self) -> list[FakeTaskConfig]:
            return [FakeTaskConfig()]

        def is_local(self) -> bool:
            return False

    captured: dict[str, Any] = {}

    class FakeTaskClient:
        async def download_tasks(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            output_dir = kwargs["output_dir"]
            return SimpleNamespace(paths=[output_dir / "task"])

    monkeypatch.setattr(
        serve_module,
        "_dataset_config_for_source",
        lambda source: FakeDatasetConfig(),
    )
    monkeypatch.setattr("harbor.tasks.client.TaskClient", FakeTaskClient)
    monkeypatch.setattr("platformdirs.user_cache_path", lambda name: tmp_path)

    resolved = await serve_module._materialize_harbor_dataset("acme/dataset@v1")

    assert resolved.parent == tmp_path / "harbor-datasets"
    assert captured["task_ids"] == [task_id]
    assert captured["output_dir"] == resolved
    assert captured["export"] is True
