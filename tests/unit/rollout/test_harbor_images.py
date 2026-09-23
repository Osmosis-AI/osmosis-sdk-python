from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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
    assert environment.task_names == ("one", "two")
    assert environment.image.image.startswith(
        "us-west1-docker.pkg.dev/acme/images/harbor:osmosis--"
    )
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
        image_base="example.com/acme/harbor:osmosis",
        image="example.com/acme/harbor:osmosis--abc",
        build_args={"VERSION": "1"},
        platform="linux/amd64",
    )
    captured: dict[str, object] = {}

    async def fake_ensure(**kwargs: object) -> str:
        captured.update(kwargs)
        captured["selected_builder"] = os.environ.get("BUILDX_BUILDER")
        return request.image

    monkeypatch.setattr(
        "harbor.environments.docker.utils.ensure_docker_image_built", fake_ensure
    )
    monkeypatch.setenv("BUILDX_BUILDER", "previous")

    result = await images.BuildxImageBuilder(builder="remote").build_and_push(request)

    assert result.image == request.image
    assert captured["docker_name"] == request.image_base
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
        image_base="us-west1-docker.pkg.dev/acme/repo/harbor:osmosis",
        image="us-west1-docker.pkg.dev/acme/repo/harbor:osmosis--abc",
        build_args={"VERSION": "1"},
        platform="linux/amd64",
    )
    submitted: dict[str, object] = {}

    async def missing_image(image: str, *, project: str) -> str | None:
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
    assert submitted["command"][:3] == ["gcloud", "builds", "submit"]
    assert "--async" in submitted["command"]
    assert "--suppress-logs" in submitted["command"]
    assert (
        submitted["command"][submitted["command"].index("--gcs-source-staging-dir") + 1]
        == "gs://acme_cloudbuild/source"
    )
    config = submitted["config"]
    assert isinstance(config, dict)
    assert config["images"] == [request.image]
    step = config["steps"][0]
    assert "--build-arg=VERSION=1" in step["args"]


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
        images._image_base(repository)
