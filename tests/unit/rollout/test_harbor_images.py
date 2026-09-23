from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import toml
from harbor.environments.definition import environment_content_hash

from osmosis_ai.rollout.backend.harbor import images
from osmosis_ai.rollout.backend.harbor.dataset import ResolvedHarborDataset


def make_task(root: Path, name: str, dockerfile: str = "FROM python:3.12\n") -> Path:
    task = root / name
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(dockerfile)
    (task / "instruction.md").write_text("Do the task.\n")
    (task / "task.toml").write_text(f'[task]\nname = "test/{name}"\n')
    return task


def test_prebuilt_image_reference_uses_harbor_environment_hash(tmp_path: Path) -> None:
    task = make_task(tmp_path, "one")
    expected_hash = environment_content_hash(task / "environment")

    assert (
        images.prebuilt_image_reference(
            "us-west1-docker.pkg.dev/acme/repo/harbor/",
            task / "environment",
        )
        == f"us-west1-docker.pkg.dev/acme/repo/harbor:{expected_hash}"
    )


def test_configure_task_prebuilt_image_only_changes_materialized_copy(
    tmp_path: Path,
) -> None:
    source = make_task(tmp_path, "source")
    materialized = tmp_path / "materialized"
    shutil.copytree(source, materialized)

    image = images.configure_task_prebuilt_image(
        materialized,
        "example.com/acme/harbor",
    )

    assert "docker_image" not in toml.load(source / "task.toml").get("environment", {})
    assert toml.load(materialized / "task.toml")["environment"]["docker_image"] == image
    assert image == images.prebuilt_image_reference(
        "example.com/acme/harbor",
        source / "environment",
    )


async def test_build_and_publish_deduplicates_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tasks = tmp_path / "tasks"
    first = make_task(tasks, "one")
    second = make_task(tasks, "two")

    async def fake_resolve(*args: object, **kwargs: object) -> ResolvedHarborDataset:
        assert args == ("https://github.com/acme/tasks.git",)
        assert kwargs == {"disable_verification": True}
        return ResolvedHarborDataset(tasks, (first, second))

    requests: list[images.BuildRequest] = []

    class FakeBuilder:
        async def build_and_push(
            self, request: images.BuildRequest
        ) -> images.PublishedImage:
            requests.append(request)
            assert not (request.context / "osmosis-requirements.txt").exists()
            assert request.dockerfile.read_text() == "FROM python:3.12\n"
            return images.PublishedImage(request.image, "sha256:abc")

    monkeypatch.setattr(images, "resolve_harbor_dataset", fake_resolve)
    monkeypatch.setattr(
        images, "_select_builder", lambda *args, **kwargs: FakeBuilder()
    )

    result = await images.build_and_publish(
        "https://github.com/acme/tasks.git",
        image_repository="us-west1-docker.pkg.dev/acme/images/harbor",
        build_system="google-cloud-build",
        gcp_project="acme",
        gcp_region="us-west1",
    )

    assert result.task_count == 2
    assert len(requests) == 1
    assert len(result.environments) == 1
    environment = result.environments[0]
    expected_hash = environment_content_hash(first / "environment")
    assert environment.content_hash == expected_hash
    assert environment.task_names == ("test/one", "test/two")
    assert environment.image.image == (
        f"us-west1-docker.pkg.dev/acme/images/harbor:{expected_hash}"
    )
    assert requests[0].context == first / "environment"
    assert requests[0].image == environment.image.image
    assert "uv venv" not in (first / "environment" / "Dockerfile").read_text()


async def test_build_and_publish_rejects_non_dockerfile_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)

    async def fake_resolve(*args: object, **kwargs: object) -> ResolvedHarborDataset:
        return ResolvedHarborDataset(tmp_path, (task,))

    monkeypatch.setattr(images, "resolve_harbor_dataset", fake_resolve)

    with pytest.raises(ValueError, match="Dockerfile environments only"):
        await images.build_and_publish(
            "./tasks",
            image_repository="example.com/acme/harbor",
        )


async def test_buildx_builder_uses_harbor_and_restores_selected_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = images.BuildRequest(
        context=tmp_path,
        dockerfile=tmp_path / "Dockerfile",
        image="example.com/acme/harbor:abc",
        platform="linux/amd64",
    )
    captured: dict[str, object] = {}

    async def missing(_image: str, logger: object = None) -> bool:
        return False

    async def fake_build(**kwargs: object) -> None:
        captured.update(kwargs)
        captured["selected_builder"] = os.environ.get("BUILDX_BUILDER")

    monkeypatch.setattr(
        "harbor.environments.docker.utils.remote_docker_image_exists", missing
    )
    monkeypatch.setattr(
        "harbor.environments.docker.utils.build_docker_image_with_buildx", fake_build
    )
    monkeypatch.setenv("BUILDX_BUILDER", "previous")

    result = await images.BuildxImageBuilder(builder="remote").build_and_push(request)

    assert result.image == request.image
    assert captured["docker_image_name"] == request.image
    assert captured["build_args"] == {}
    assert captured["push"] is True
    assert captured["selected_builder"] == "remote"
    assert os.environ["BUILDX_BUILDER"] == "previous"


async def test_google_cloud_builder_submits_generated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    request = images.BuildRequest(
        context=tmp_path,
        dockerfile=dockerfile,
        image="us-west1-docker.pkg.dev/acme/repo/harbor:abc",
        platform="linux/amd64",
    )
    submitted: dict[str, object] = {}

    async def missing_image(image: str, *, project: str) -> None:
        assert image == request.image
        assert project == "acme"
        return None

    async def fake_run(command: list[str]) -> tuple[int, str, str]:
        if command[:3] == ["gcloud", "builds", "submit"]:
            submitted["command"] = command
            config_path = Path(command[command.index("--config") + 1])
            submitted["config"] = json.loads(config_path.read_text())
            return 0, '{"id":"build-1","status":"QUEUED"}', ""
        assert command[:3] == ["gcloud", "builds", "describe"]
        return (
            0,
            json.dumps(
                {
                    "status": "SUCCESS",
                    "results": {
                        "images": [{"name": request.image, "digest": "sha256:def"}]
                    },
                }
            ),
            "build logs",
        )

    monkeypatch.setattr(images, "_google_image_digest", missing_image)
    monkeypatch.setattr(images, "_run_command", fake_run)

    result = await images.GoogleCloudBuildImageBuilder(
        project="acme", region="us-west1"
    ).build_and_push(request)

    assert result.immutable_image.endswith("@sha256:def")
    command = submitted["command"]
    assert isinstance(command, list)
    assert command[:3] == ["gcloud", "builds", "submit"]
    assert "--async" in command
    assert "--suppress-logs" in command
    assert command[command.index("--gcs-source-staging-dir") + 1] == (
        "gs://acme_cloudbuild/source"
    )
    assert "--timeout=2400" in command
    config = submitted["config"]
    assert isinstance(config, dict)
    assert config["images"] == [request.image]
    assert len(config["steps"]) == 1
    build_step = config["steps"][0]
    assert f"--tag={request.image}" in build_step["args"]
    assert "--build-arg" not in build_step["args"]


async def test_google_artifact_registry_lookup_uses_tag_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "us-west1-docker.pkg.dev/acme/repo/harbor:abc"

    async def fake_run(command: list[str]) -> tuple[int, str, str]:
        assert command[:5] == [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "list",
        ]
        return (
            0,
            json.dumps(
                [
                    {
                        "package": "us-west1-docker.pkg.dev/acme/repo/harbor",
                        "tags": ["abc"],
                        "version": "sha256:def",
                    }
                ]
            ),
            "",
        )

    monkeypatch.setattr(images, "_run_command", fake_run)

    assert await images._google_image_digest(image, project="acme") == "sha256:def"


@pytest.mark.parametrize(
    "repository",
    ["", "example.com/acme/image:tag", "example.com/acme/image@sha256:abc"],
)
def test_image_repository_must_not_include_tag_or_digest(repository: str) -> None:
    with pytest.raises(ValueError):
        images.normalize_image_repository(repository)
