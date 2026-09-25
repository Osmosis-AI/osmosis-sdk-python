import asyncio
import io
import tarfile
import tomllib
from types import SimpleNamespace

import httpx
import pytest
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

ORG = "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12
SOURCE = TaskSource("https://github.com/Acme/Tasks.git", "tasks", "a" * 40)
IMAGE = "us-west1-docker.pkg.dev/project/repo/environment@sha256:" + "d" * 64


from tests.unit.harbor_helpers import make_task


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


def test_builder_and_gateway_fetch_identical_pinned_archives(
    tmp_path, monkeypatch, caplog
):
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


async def test_allow_empty_does_not_accept_an_empty_explicit_dataset(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "dataset.toml").write_text('[dataset]\nname = "acme/data"\n')
    with pytest.raises(ValueError, match="uniquely named tasks"):
        await materialize_source_tasks(
            repository, "dataset.toml", tmp_path / "tasks", allow_empty=True
        )


def test_task_source_normalizes_full_sha_case():
    assert TaskSource(SOURCE.repository, SOURCE.path, "aB" * 20).revision == "ab" * 20


def test_step_verifiers_hash_their_context_and_preserve_inheritance(tmp_path):
    from harbor.models.task.paths import TaskPaths

    task = make_task(tmp_path / "task", separate=True)
    (task / "task.toml").write_text(
        (task / "task.toml").read_text()
        + '\n[[steps]]\nname = "inherited"\n'
        + '[[steps]]\nname = "explicit"\n[steps.verifier.environment]\ncpus = 7\n'
        + '[[steps]]\nname = "shared"\n[steps.verifier]\nenvironment_mode = "shared"\n'
    )
    context = TaskPaths(task).step_tests_dir("explicit")
    context.mkdir(parents=True)
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nRUN echo verifier\n")
    identities = task_identities(task)
    assert set(identities) == {
        "environment",
        "steps.0.verifier.environment",
        "steps.1.verifier.environment",
    }
    assert identities["steps.0.verifier.environment"] == identities["environment"]
    assert identities["steps.1.verifier.environment"] != identities["environment"]
    dockerfile.write_text("FROM ubuntu:24.04\nRUN echo changed\n")
    changed = task_identities(task)
    assert {key for key in changed if changed[key] != identities[key]} == {
        "steps.1.verifier.environment"
    }

    bindings = {key: IMAGE for key in identities}
    bind_task_images(task, bindings)
    raw = tomllib.loads((task / "task.toml").read_text())
    assert raw["steps"][0]["verifier"]["environment"]["cpus"] == 3
    assert raw["steps"][1]["verifier"]["environment"]["cpus"] == 7
    assert raw["steps"][1]["verifier"]["environment"]["docker_image"] == IMAGE
    assert "environment" not in raw["steps"][2]["verifier"]


@pytest.mark.parametrize("kind", ["traversal", "expanded-size", "multiple-roots"])
def test_source_fetch_rejects_unsafe_archives(tmp_path, monkeypatch, kind):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if kind == "multiple-roots":
            for name in ("first", "second"):
                entry = tarfile.TarInfo(name)
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
        else:
            entry = tarfile.TarInfo("../escape" if kind == "traversal" else "root/data")
            contents = b"x" * 4096
            entry.size = len(contents)
            archive.addfile(entry, io.BytesIO(contents))
    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=buffer.getvalue())
            ),
            **kwargs,
        ),
    )
    if kind == "expanded-size":
        monkeypatch.setattr("osmosis_ai.harbor_images.MAX_SOURCE_BYTES", 1024)
    with pytest.raises((ValueError, tarfile.FilterError)):
        fetch_source(SOURCE, "installation-token", tmp_path / "source")
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "escape").exists()


async def test_empty_task_discovery_requires_explicit_opt_in(tmp_path):
    repository = tmp_path / "runtime"
    repository.mkdir()
    (repository / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ValueError, match="no Harbor tasks"):
        await materialize_source_tasks(repository, ".", tmp_path / "default")
    assert (
        await materialize_source_tasks(
            repository, ".", tmp_path / "allowed", allow_empty=True
        )
        == []
    )
