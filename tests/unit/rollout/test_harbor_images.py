from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from harbor.environments.definition import environment_content_hash
from harbor.environments.gke import GKEEnvironment

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


def test_image_reference_matches_harbor_gke_contract() -> None:
    environment = object.__new__(GKEEnvironment)
    environment.registry_location = "us-west1"
    environment.project_id = "acme"
    environment.registry_name = "harbor-sandbox"
    environment.environment_name = "multiply-0000"
    environment.__dict__["environment_id"] = "abc123"

    assert images._harbor_image_reference(
        "us-west1-docker.pkg.dev/acme/harbor-sandbox",
        environment_name="multiply-0000",
        environment_id="abc123",
    ) == environment._get_image_url()


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
        ) -> tuple[images.PublishedImage, ...]:
            requests.append(request)
            assert not (request.context / "osmosis-requirements.txt").exists()
            assert request.dockerfile.read_text() == "FROM python:3.12\n"
            return tuple(
                images.PublishedImage(image, "sha256:abc")
                for image in request.images
            )

    monkeypatch.setattr(images, "resolve_harbor_dataset", fake_resolve)
    monkeypatch.setattr(
        images, "_select_builder", lambda *args, **kwargs: FakeBuilder()
    )

    result = await images.build_and_publish(
        "https://github.com/acme/tasks.git",
        image_repository="us-west1-docker.pkg.dev/acme/images",
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
    assert tuple(image.image for image in environment.images) == (
        f"us-west1-docker.pkg.dev/acme/images/one:{expected_hash}",
        f"us-west1-docker.pkg.dev/acme/images/two:{expected_hash}",
    )
    assert requests[0].context == first / "environment"
    assert requests[0].images == tuple(image.image for image in environment.images)
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
        images=(
            "example.com/acme/one:abc",
            "example.com/acme/two:abc",
        ),
        platform="linux/amd64",
    )
    captured: dict[str, object] = {}

    async def missing(_image: str, logger: object = None) -> bool:
        return False

    async def fake_build(**kwargs: object) -> None:
        captured.update(kwargs)
        captured["selected_builder"] = os.environ.get("BUILDX_BUILDER")

    async def fake_run(command: list[str]) -> tuple[int, str, str]:
        captured["alias_command"] = command
        return 0, "", ""

    monkeypatch.setattr(
        "harbor.environments.docker.utils.remote_docker_image_exists", missing
    )
    monkeypatch.setattr(
        "harbor.environments.docker.utils.build_docker_image_with_buildx", fake_build
    )
    monkeypatch.setattr(images, "_run_command", fake_run)
    monkeypatch.setenv("BUILDX_BUILDER", "previous")

    result = await images.BuildxImageBuilder(builder="remote").build_and_push(request)

    assert tuple(image.image for image in result) == request.images
    assert captured["docker_image_name"] == request.images[0]
    assert captured["build_args"] == {}
    assert captured["push"] is True
    assert captured["selected_builder"] == "remote"
    assert captured["alias_command"] == [
        "docker",
        "buildx",
        "imagetools",
        "create",
        "--tag",
        request.images[1],
        request.images[0],
    ]
    assert os.environ["BUILDX_BUILDER"] == "previous"


async def test_google_cloud_builder_submits_generated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    request = images.BuildRequest(
        context=tmp_path,
        dockerfile=dockerfile,
        images=(
            "us-west1-docker.pkg.dev/acme/repo/one:abc",
            "us-west1-docker.pkg.dev/acme/repo/two:abc",
        ),
        platform="linux/amd64",
    )
    submitted: dict[str, object] = {}

    async def missing_images(
        image_names: tuple[str, ...], *, project: str
    ) -> dict[str, str]:
        assert project == "acme"
        return {}

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
                        "images": [
                            {"name": image, "digest": "sha256:def"}
                            for image in request.images
                        ]
                    },
                }
            ),
            "build logs",
        )

    monkeypatch.setattr(images, "_google_image_digests", missing_images)
    monkeypatch.setattr(images, "_run_command", fake_run)

    result = await images.GoogleCloudBuildImageBuilder(
        project="acme", region="us-west1"
    ).build_and_push(request)

    assert all(image.immutable_image.endswith("@sha256:def") for image in result)
    assert submitted["command"][:3] == ["gcloud", "builds", "submit"]
    assert "--async" in submitted["command"]
    assert "--suppress-logs" in submitted["command"]
    assert (
        submitted["command"][submitted["command"].index("--gcs-source-staging-dir") + 1]
        == "gs://acme_cloudbuild/source"
    )
    assert "--timeout=2400" in submitted["command"]
    config = submitted["config"]
    assert isinstance(config, dict)
    assert config["images"] == [request.images[0]]
    build_step, alias_step = config["steps"]
    assert f"--tag={request.images[0]}" in build_step["args"]
    assert "--build-arg" not in build_step["args"]
    assert alias_step["entrypoint"] == "sh"
    assert alias_step["args"][0] == "-ceu"
    assert f"docker tag {request.images[0]} {request.images[1]}" in alias_step["args"][1]
    assert f"docker push {request.images[1]}" in alias_step["args"][1]


def test_google_cloud_config_batches_large_harbor_datasets(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    image_names = tuple(
        f"us-west1-docker.pkg.dev/acme/repo/multiply-{index:04d}:abc123"
        for index in range(300)
    )

    config = images._cloud_build_config(
        images.BuildRequest(
            context=tmp_path,
            dockerfile=dockerfile,
            images=image_names,
            platform="linux/amd64",
        )
    )

    assert config["images"] == [image_names[0]]
    assert len(config["steps"]) > 2
    for step in config["steps"]:
        assert len(step["args"]) <= 100
        assert all(len(argument) <= 10_000 for argument in step["args"])
    alias_scripts = "\n".join(step["args"][1] for step in config["steps"][1:])
    assert sum(f"docker push {image}" in alias_scripts for image in image_names[1:]) == 299


async def test_google_artifact_registry_lookup_uses_tag_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "us-west1-docker.pkg.dev/acme/repo/harbor:osmosis--abc"

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
                        "tags": ["osmosis--abc"],
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
        images._image_repository(repository)
