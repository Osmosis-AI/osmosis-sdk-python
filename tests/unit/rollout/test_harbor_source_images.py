import asyncio
import hashlib
import json
import tomllib
from dataclasses import asdict
from unittest.mock import Mock

import httpx
import pytest
from harbor.trial.queue import TrialQueue

from osmosis_ai.harbor_images import (
    TaskSource,
    materialize_source_tasks,
)
from osmosis_ai.rollout.backend.harbor.source import (
    RegistryResolver,
    SourceHarborBackend,
)
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask

ORG = "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12
SOURCE = TaskSource("https://github.com/Acme/Tasks.git", "tasks", "a" * 40)
IMAGE = "us-west1-docker.pkg.dev/project/repo/environment@sha256:" + "d" * 64


from tests.unit.harbor_helpers import make_task


def test_discovery_and_trial_binding_preserve_original_source(tmp_path):
    original = make_task(tmp_path / "repository/tasks/group/add", separate=True)
    before = (original / "task.toml").read_bytes()
    destination = tmp_path / "download"
    assert asyncio.run(
        materialize_source_tasks(tmp_path / "repository", "tasks", destination)
    ) == ["group/add"]
    backend = SourceHarborBackend(
        tasks_dir=destination,
        orchestrator=TrialQueue(n_concurrent=1),
        agent="opencode",
        task_mode="dataset",
        image_bindings={
            "group/add": {"environment": IMAGE, "verifier.environment": IMAGE}
        },
    )
    backend.rollouts_dir = tmp_path / "rollouts"
    trial = backend.prewarm_trial_config(HarborTask(destination / "group/add"))
    raw = tomllib.loads((trial.task.path / "task.toml").read_text())
    assert raw["environment"]["docker_image"] == IMAGE
    assert raw["verifier"]["environment"]["docker_image"] == IMAGE
    assert raw["verifier"]["environment"]["cpus"] == 3
    assert trial.install_only is True
    assert (original / "task.toml").read_bytes() == before
    assert (destination / "group/add/task.toml").read_bytes() == before


@pytest.mark.parametrize(
    "config",
    [
        "malformed local credentials",
        json.dumps({"credHelpers": {"us-west1-docker.pkg.dev": "gcloud"}}),
    ],
)
def test_registry_resolves_verifies_and_caches_manifest(tmp_path, monkeypatch, config):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text(config)
    content = b'{"schemaVersion":2}'
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    response = httpx.Response(
        200, content=content, headers={"docker-content-digest": digest}
    )
    client = Mock()
    client.get.return_value = response
    context = Mock(
        __enter__=Mock(return_value=client), __exit__=Mock(return_value=False)
    )
    monkeypatch.setattr(httpx, "Client", Mock(return_value=context))
    resolver = RegistryResolver("us-west1-docker.pkg.dev/project/repo", "private-token")
    assert (
        resolver.resolve("tag")
        == resolver.resolve("tag")
        == "us-west1-docker.pkg.dev/project/repo/environment@" + digest
    )
    assert client.get.call_count == 1
    client.get.return_value = httpx.Response(404)
    with pytest.raises(RuntimeError, match="images build"):
        resolver.resolve("missing")
    for status in (401, 403):
        client.get.return_value = httpx.Response(status)
        with pytest.raises(RuntimeError, match="Registry authentication failed"):
            resolver.resolve("unauthorized")
    client.get.return_value = httpx.Response(
        200, content=content, headers={"docker-content-digest": "sha256:" + "b" * 64}
    )
    with pytest.raises(ValueError, match="verification"):
        resolver.resolve("corrupt")


@pytest.mark.parametrize(
    "config",
    [
        "invalid JSON",
        "[]",
        json.dumps({"auths": {"us-west1-docker.pkg.dev": {"auth": "not base64!"}}}),
        json.dumps({"auths": {"us-west1-docker.pkg.dev": {"auth": "bm9jb2xvbg=="}}}),
    ],
)
def test_malformed_docker_credentials_fail_cleanly(tmp_path, monkeypatch, config):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text(config)
    client = Mock()
    monkeypatch.setattr(httpx, "Client", client)
    resolver = RegistryResolver("us-west1-docker.pkg.dev/project/repo")
    with pytest.raises(
        RuntimeError, match="Registry credentials could not be read"
    ) as error:
        resolver.resolve("tag")
    assert config not in str(error.value)
    client.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {"credHelpers": {"us-west1-docker.pkg.dev": "gcloud"}},
        {"credsStore": "desktop"},
        {
            "credHelpers": {"us-west1-docker.pkg.dev": "gcloud"},
            "auths": {"us-west1-docker.pkg.dev": {"auth": "dXNlcjpwYXNz"}},
        },
    ],
)
def test_registry_rejects_unsupported_docker_helpers_before_request(
    tmp_path, monkeypatch, config
):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(config))
    client = Mock()
    monkeypatch.setattr(httpx, "Client", client)
    resolver = RegistryResolver("us-west1-docker.pkg.dev/project/repo")
    with pytest.raises(
        RuntimeError, match=r"Docker credential helpers.*managed source gateway"
    ):
        resolver.resolve("tag")
    client.assert_not_called()


def test_gateway_readiness_preserves_source_and_verifier_healthcheck(tmp_path):
    task = make_task(tmp_path / "tasks/add", separate=True)
    path = task / "task.toml"
    path.write_text(
        path.read_text()
        + '\n[environment.healthcheck]\ncommand = "test -f /task-ready"\ntimeout_sec = 12\n'
    )
    before = path.read_bytes()
    backend = SourceHarborBackend(
        tasks_dir=tmp_path / "tasks",
        task_mode="dataset",
        agent="opencode",
        orchestrator=TrialQueue(n_concurrent=1),
        image_bindings={"add": {"environment": IMAGE, "verifier.environment": IMAGE}},
        environment_healthcheck={
            "command": "test -f /route-ready",
            "timeout_sec": 5,
            "retries": 30,
        },
    )
    backend.rollouts_dir = tmp_path / "rollouts"
    trial = backend.prewarm_trial_config(HarborTask(task))
    raw = tomllib.loads((trial.task.path / "task.toml").read_text())
    assert (
        raw["environment"]["healthcheck"]["command"]
        == "test -f /route-ready && (test -f /task-ready)"
    )
    assert raw["environment"]["healthcheck"]["timeout_sec"] == 17
    assert raw["environment"]["healthcheck"]["retries"] == 30
    assert (
        raw["verifier"]["environment"]["healthcheck"]["command"]
        == "test -f /task-ready"
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("domain", "overrides", "expected"),
    [
        (
            "http://opensandbox.internal",
            {},
            {"protocol": "http", "use_server_proxy": True},
        ),
        (
            "https://opensandbox.example.com",
            {},
            {"protocol": "https", "use_server_proxy": True},
        ),
        (
            "opensandbox.example.com",
            {},
            {"protocol": "https", "use_server_proxy": True},
        ),
        (
            "",
            {"domain": "http://custom.internal"},
            {
                "domain": "http://custom.internal",
                "protocol": "http",
                "use_server_proxy": True,
            },
        ),
        (
            "http://opensandbox.internal",
            {"protocol": "https", "use_server_proxy": False},
            {"protocol": "https", "use_server_proxy": False},
        ),
    ],
)
def test_source_gateway_transport_matches_service_and_preserves_overrides(
    monkeypatch, domain, overrides, expected
):
    from osmosis_ai.rollout.backend.harbor import source

    monkeypatch.setenv("OPENSANDBOX_DOMAIN", domain)
    monkeypatch.setenv(
        "_OSMOSIS_HARBOR_CONFIG", json.dumps({"environment_kwargs": overrides})
    )
    monkeypatch.setenv("_OSMOSIS_HARBOR_TASK_SOURCE", json.dumps(asdict(SOURCE)))
    monkeypatch.setenv("_OSMOSIS_ORGANIZATION_ID", ORG)
    monkeypatch.setenv(
        "_OSMOSIS_HARBOR_REGISTRY",
        "us-west1-docker.pkg.dev/project/" + SOURCE.repository_id(ORG),
    )
    monkeypatch.setattr(
        source,
        "fetch_source",
        lambda _source, _token, root: make_task(root / "tasks/add"),
    )
    monkeypatch.setattr(source.RegistryResolver, "resolve", lambda *_: IMAGE)
    backend = Mock()
    monkeypatch.setattr(source, "SourceHarborBackend", backend)
    monkeypatch.setattr("osmosis_ai.rollout.server.create_rollout_server", Mock())
    monkeypatch.setattr("uvicorn.run", Mock())

    source.main()

    assert backend.call_args.kwargs["environment_config"].kwargs == expected
