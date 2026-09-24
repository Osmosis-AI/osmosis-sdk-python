import asyncio
import hashlib
import io
import json
import tarfile
import tomllib
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from harbor.trial.queue import TrialQueue
from harbor.utils.container_cache import docker_build_context_hash

from osmosis_ai.harbor_images import (
    PLATFORM,
    TaskSource,
    bind_task_images,
    environment_identity,
    fetch_source,
    materialize_source_tasks,
    normalized_context,
    task_identities,
)
from osmosis_ai.platform.cli.dev_server import up_source
from osmosis_ai.rollout.backend.harbor.source import (
    RegistryResolver,
    SourceHarborBackend,
)
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask

ORG = "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12
SOURCE = TaskSource("https://github.com/Acme/Tasks.git", "tasks", "a" * 40)
IMAGE = "us-west1-docker.pkg.dev/project/repo/environment@sha256:" + "d" * 64


def make_task(path, separate=False):
    (path / "environment").mkdir(parents=True)
    (path / "environment/Dockerfile").write_text("FROM ubuntu:24.04\nCOPY data /data\n")
    (path / "environment/data").write_text("content")
    (path / "instruction.md").write_text("Do something")
    (path / "tests").mkdir()
    (path / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (path / "task.toml").write_text(
        "[environment]\ncpus = 3\n"
        + ('[verifier]\nenvironment_mode = "separate"\n' if separate else "")
    )
    return path


def test_git_archive_modes_and_harbor_hash_agree(tmp_path):
    task = make_task(tmp_path / "task")
    context = task / "environment"
    (context / "data").chmod(0o600)
    with normalized_context(context) as normalized:
        first = environment_identity(normalized)
        assert first.environment_hash == docker_build_context_hash(
            context=normalized, platform=PLATFORM, build_args={}
        )
        assert (normalized / "data").stat().st_mode & 0o777 == 0o644
    assert (context / "data").stat().st_mode & 0o777 == 0o600
    (context / "data").chmod(0o644)
    assert task_identities(task)["environment"] == first
    (context / "data").chmod(0o755)
    assert task_identities(task)["environment"] != first


def test_only_environment_changes_invalidate_images(tmp_path):
    task = make_task(tmp_path / "task", separate=True)
    first = task_identities(task)
    assert first["environment"] == first["verifier.environment"]
    (task / "instruction.md").write_text("Different instruction")
    (task / "tests/test.sh").write_text("different verifier script")
    assert task_identities(task) == first
    (task / "task.toml").write_text(
        '[environment]\ncpus = 3\n[verifier]\nenvironment_mode = "separate"\n[verifier.environment]\ndocker_image = "ubuntu:24.04"\n'
    )
    separate = task_identities(task)
    assert separate["environment"] == first["environment"]
    assert separate["verifier.environment"] != first["environment"]
    (task / "environment/data").write_text("different build input")
    assert task_identities(task)["environment"] != first["environment"]


def test_repo_namespace_is_canonical_collision_resistant_and_workspace_scoped():
    assert SOURCE.repository == "https://github.com/acme/tasks"
    assert SOURCE.repository_id(ORG) == TaskSource(
        SOURCE.repository, "tasks", "b" * 40
    ).repository_id(ORG)
    assert SOURCE.repository_id(ORG) != SOURCE.repository_id("f" * 8 + ORG[8:])
    assert TaskSource(SOURCE.repository, "a/b", SOURCE.revision).repository_id(
        ORG
    ) != TaskSource(SOURCE.repository, "a-b", SOURCE.revision).repository_id(ORG)
    assert len(SOURCE.repository_id(ORG)) <= 63


@pytest.mark.parametrize(
    "path", ["/tasks", "../tasks", "tasks//one", "tasks\\one", "tasks\n"]
)
def test_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        TaskSource(SOURCE.repository, path, SOURCE.revision)


def test_source_requires_immutable_revision():
    with pytest.raises(ValueError, match="full Git commit"):
        TaskSource(SOURCE.repository, "tasks", "main")


def test_repository_root_task_id_does_not_depend_on_checkout_name(tmp_path):
    for name in ("build-checkout", "gateway-checkout"):
        make_task(tmp_path / name)
        assert asyncio.run(
            materialize_source_tasks(tmp_path / name, ".", tmp_path / (name + "-tasks"))
        ) == ["task"]


def test_builder_and_gateway_fetch_identical_pinned_archives(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="httpx")
    task = make_task(tmp_path / "input/tasks/add")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(tmp_path / "input", arcname="acme-tasks-commit")
    requested = []

    def handle(request):
        requested.append(request)
        if request.url.host == "api.github.com":
            assert request.url.path.endswith("/tarball/" + SOURCE.revision)
            return httpx.Response(
                302,
                headers={
                    "location": "https://codeload.github.com/acme/tasks/legacy.tar.gz/"
                    + SOURCE.revision
                    + "?token=private-download-token"
                },
            )
        assert request.url.host == "codeload.github.com"
        assert request.url.query == b""
        assert request.headers["authorization"] == "Bearer private-installation-token"
        return httpx.Response(200, content=buffer.getvalue())

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    for name in ("builder", "gateway"):
        fetch_source(SOURCE, "private-installation-token", tmp_path / name)
        assert task_identities(tmp_path / name / "tasks/add") == task_identities(task)
    assert len(requested) == 4
    assert "private-download-token" not in caplog.text
    assert "private-installation-token" not in caplog.text


@pytest.mark.parametrize(
    "location",
    [
        "http://codeload.github.com/task",
        "https://evil.test/task",
        "https://codeload.github.com:secret/task",
    ],
)
def test_source_fetch_rejects_untrusted_redirect_without_exposing_credentials(
    tmp_path, monkeypatch, location
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        fetch_source(SOURCE, "private-installation-token", tmp_path / "source")
    assert "secret" not in str(error.value)
    assert "private-installation-token" not in str(error.value)
    assert len(requests) == 1


def test_source_fetch_bounds_download_size(tmp_path, monkeypatch):
    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=b"too large")
            ),
            **kwargs,
        ),
    )
    monkeypatch.setattr("osmosis_ai.harbor_images.MAX_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        fetch_source(SOURCE, "token", tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_gateway_rejects_submodules_like_the_builder(tmp_path):
    make_task(tmp_path / "repository/tasks/add")
    (tmp_path / "repository/.gitmodules").write_text("[submodule]")
    with pytest.raises(ValueError, match="submodules"):
        asyncio.run(
            materialize_source_tasks(
                tmp_path / "repository", "tasks", tmp_path / "tasks"
            )
        )


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


def test_missing_verifier_binding_fails_before_trial(tmp_path):
    task = make_task(tmp_path / "task", separate=True)
    with pytest.raises(ValueError, match="every task environment"):
        bind_task_images(task, {"environment": IMAGE})


@pytest.mark.parametrize("kind", ["symlink", "lfs"])
def test_rejects_unmaterialized_sources(tmp_path, kind):
    task = make_task(tmp_path / "repo/tasks/add")
    if kind == "symlink":
        (task / "environment/link").symlink_to(tmp_path / "outside")
    else:
        (task / "environment/data").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:123"
        )
    with pytest.raises(ValueError):
        asyncio.run(
            materialize_source_tasks(tmp_path / "repo", "tasks", tmp_path / "download")
        )


async def test_hub_manifest_downloads_exact_task_digests(tmp_path, monkeypatch):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo/dataset.toml").write_text(
        '[dataset]\nname = "acme/data"\n[[tasks]]\nname = "acme/add"\ndigest = "sha256:'
        + "a" * 64
        + '"\n'
    )
    task = make_task(tmp_path / "cached-task")

    async def download(self, ids, **kwargs):
        assert ids[0].ref == "sha256:" + "a" * 64
        assert ids[0].get_name() == "acme/add"
        return SimpleNamespace(paths=[task])

    monkeypatch.setattr("harbor.tasks.client.TaskClient.download_tasks", download)
    names = await materialize_source_tasks(
        tmp_path / "repo", "dataset.toml", tmp_path / "tasks"
    )
    assert names == ["acme/add"]
    assert (tmp_path / "tasks/acme/add/task.toml").is_file()


def test_registry_resolves_verifies_and_caches_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text("malformed local credentials")
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


def test_task_source_normalizes_full_sha_case():
    assert TaskSource(SOURCE.repository, SOURCE.path, "aB" * 20).revision == "ab" * 20


def test_dev_source_up_does_not_require_local_gateway_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = Mock()
    client.provision_dev_rollout_server.return_value = {
        "id": "server",
        "url": "https://gateway.test",
        "backend": "gke",
        "sandbox_environment": "opensandbox",
        "task_source": asdict(SOURCE),
    }
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.dev_server.OsmosisClient", lambda: client
    )
    up_source(
        url=SOURCE.repository,
        path=SOURCE.path,
        ref=SOURCE.revision,
        ttl_hours=24,
        backend=None,
        sandbox_environment=None,
    )
    assert client.provision_dev_rollout_server.call_args.kwargs[
        "task_source"
    ] == asdict(SOURCE)
    assert not list(tmp_path.iterdir())


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
