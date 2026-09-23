"""Build and publish content-addressed Harbor task images."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from osmosis_ai.rollout.backend.harbor.dataset import resolve_harbor_dataset
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask

if TYPE_CHECKING:
    from harbor.models.task.task import Task


class BuildSystem(StrEnum):
    BUILDX = "buildx"
    GOOGLE_CLOUD_BUILD = "google-cloud-build"


@dataclass(frozen=True)
class BuildRequest:
    context: Path
    dockerfile: Path
    images: tuple[str, ...]
    platform: str


@dataclass(frozen=True)
class PublishedImage:
    image: str
    digest: str | None = None

    @property
    def immutable_image(self) -> str:
        if self.digest is None:
            return self.image
        repository, _, _tag = self.image.rpartition(":")
        return f"{repository}@{self.digest}"


@dataclass(frozen=True)
class PublishedEnvironment:
    content_hash: str
    images: tuple[PublishedImage, ...]
    task_names: tuple[str, ...]


@dataclass(frozen=True)
class BuildAndPublishResult:
    dataset: str
    dataset_path: Path
    task_count: int
    environments: tuple[PublishedEnvironment, ...]


class ImageBuilder(Protocol):
    async def build_and_push(
        self, request: BuildRequest
    ) -> tuple[PublishedImage, ...]: ...


class BuildxImageBuilder:
    """Publish Harbor-compatible image references with Buildx."""

    def __init__(self, *, builder: str | None = None) -> None:
        self.builder = builder

    async def build_and_push(
        self, request: BuildRequest
    ) -> tuple[PublishedImage, ...]:
        from harbor.environments.docker.utils import (
            build_docker_image_with_buildx,
            remote_docker_image_exists,
        )

        if not request.images:
            return ()
        primary, *aliases = request.images

        previous_builder = os.environ.get("BUILDX_BUILDER")
        if self.builder is not None:
            os.environ["BUILDX_BUILDER"] = self.builder
        try:
            if not await remote_docker_image_exists(primary):
                with tempfile.TemporaryDirectory(
                    prefix="osmosis-harbor-buildx-"
                ) as temp:
                    await build_docker_image_with_buildx(
                        docker_image_name=primary,
                        context=request.context,
                        dockerfile_path=request.dockerfile,
                        build_log_path=Path(temp) / "build.log",
                        build_args={},
                        platform=request.platform,
                        push=True,
                    )

            if aliases:
                command = ["docker", "buildx", "imagetools", "create"]
                for alias in aliases:
                    command.extend(("--tag", alias))
                command.append(primary)
                returncode, output, error = await _run_command(command)
                if returncode != 0:
                    details = error.strip() or output.strip()
                    raise RuntimeError(
                        f"Failed to publish Harbor image aliases for {primary}: "
                        f"{details}"
                    )
        finally:
            if self.builder is not None:
                if previous_builder is None:
                    os.environ.pop("BUILDX_BUILDER", None)
                else:
                    os.environ["BUILDX_BUILDER"] = previous_builder

        return tuple(PublishedImage(image=image) for image in request.images)


async def _run_command(command: Sequence[str]) -> tuple[int, str, str]:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command is not installed: {command[0]}") from exc

    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _google_image_lookup_command(image: str, *, project: str) -> list[str] | None:
    registry = image.split("/", 1)[0]
    if registry == "gcr.io" or registry.endswith(".gcr.io"):
        return [
            "gcloud",
            "container",
            "images",
            "describe",
            image,
            "--project",
            project,
            "--format=value(image_summary.digest)",
        ]
    if registry.endswith("-docker.pkg.dev"):
        image_path, separator, tag = image.rpartition(":")
        if not separator:
            return None
        return [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "list",
            image_path,
            "--include-tags",
            f"--filter=tags:{tag}",
            "--project",
            project,
            "--format=json",
        ]
    return None


async def _google_image_digest(image: str, *, project: str) -> str | None:
    command = _google_image_lookup_command(image, project=project)
    if command is None:
        return None
    returncode, output, _error = await _run_command(command)
    if returncode != 0:
        return None
    if "artifacts" in command:
        image_path, _, tag = image.rpartition(":")
        try:
            results = json.loads(output)
        except json.JSONDecodeError:
            return None
        for result in results:
            if result.get("package") == image_path and tag in result.get("tags", []):
                digest = str(result.get("version", ""))
                return digest if digest.startswith("sha256:") else None
        return None
    digest = output.strip()
    return digest if digest.startswith("sha256:") else None


def _artifact_registry_root(image: str) -> str | None:
    parts = image.split("/")
    if len(parts) < 4 or not parts[0].endswith("-docker.pkg.dev"):
        return None
    return "/".join(parts[:3])


async def _google_image_digests(
    image_names: Sequence[str], *, project: str
) -> dict[str, str]:
    """Read a Harbor environment's tags with one Artifact Registry request."""
    if not image_names:
        return {}

    roots = {_artifact_registry_root(image) for image in image_names}
    tags = {image.rpartition(":")[2] for image in image_names}
    if None in roots or len(roots) != 1 or len(tags) != 1:
        resolved = await asyncio.gather(
            *(_google_image_digest(image, project=project) for image in image_names)
        )
        return {
            image: digest
            for image, digest in zip(image_names, resolved, strict=True)
            if digest is not None
        }

    root = roots.pop()
    assert root is not None
    tag = tags.pop()
    command = [
        "gcloud",
        "artifacts",
        "docker",
        "images",
        "list",
        root,
        "--include-tags",
        f"--filter=tags:{tag}",
        "--project",
        project,
        "--format=json",
    ]
    returncode, output, _error = await _run_command(command)
    if returncode != 0:
        return {}
    try:
        results = json.loads(output)
    except json.JSONDecodeError:
        return {}

    expected = set(image_names)
    digests: dict[str, str] = {}
    for result in results:
        package = result.get("package")
        digest = str(result.get("version", ""))
        if not isinstance(package, str) or not digest.startswith("sha256:"):
            continue
        for result_tag in result.get("tags", []):
            image = f"{package}:{result_tag}"
            if image in expected:
                digests[image] = digest
    return digests


def _cloud_build_config(request: BuildRequest) -> dict[str, object]:
    dockerfile = request.dockerfile.relative_to(request.context).as_posix()
    primary, *aliases = request.images
    steps: list[dict[str, object]] = [
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": [
                "build",
                f"--file={dockerfile}",
                f"--platform={request.platform}",
                f"--tag={primary}",
                ".",
            ],
        }
    ]

    # Cloud Build limits each step to 100 arguments and each argument to 10 KB.
    # Tag and push aliases in bounded shell scripts while reusing the one image
    # built above. Artifact Registry deduplicates the shared layers.
    scripts: list[str] = []
    current: list[str] = []
    current_size = 0
    for alias in aliases:
        command = shlex.join(["docker", "tag", primary, alias]) + "\n" + shlex.join(
            ["docker", "push", alias]
        )
        if current and current_size + len(command) + 1 > 8_000:
            scripts.append("\n".join(current))
            current = []
            current_size = 0
        current.append(command)
        current_size += len(command) + 1
    if current:
        scripts.append("\n".join(current))

    steps.extend(
        {
            "name": "gcr.io/cloud-builders/docker",
            "entrypoint": "sh",
            "args": ["-ceu", script],
        }
        for script in scripts
    )
    return {
        "steps": steps,
        "images": [primary],
    }


def _cloud_build_digests(output: str) -> dict[str, str]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {}
    images = payload.get("results", {}).get("images", [])
    digests: dict[str, str] = {}
    for result in images:
        name = result.get("name")
        digest = result.get("digest")
        if (
            isinstance(name, str)
            and isinstance(digest, str)
            and digest.startswith("sha256:")
        ):
            digests[name] = digest
    return digests


def _cloud_build_payload(output: str) -> dict[str, object]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Google Cloud Build returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Google Cloud Build returned an invalid response")
    return payload


class GoogleCloudBuildImageBuilder:
    """Submit image builds to Google Cloud Build."""

    def __init__(self, *, project: str, region: str) -> None:
        if not project.strip():
            raise ValueError("gcp_project must be non-empty")
        if not region.strip():
            raise ValueError("gcp_region must be non-empty")
        self.project = project
        self.region = region

    async def build_and_push(
        self, request: BuildRequest
    ) -> tuple[PublishedImage, ...]:
        if not request.images:
            return ()
        if any(
            _google_image_lookup_command(image, project=self.project) is None
            for image in request.images
        ):
            raise ValueError(
                "google-cloud-build currently requires a Google Artifact Registry "
                "or Google Container Registry image repository"
            )
        digests = await _google_image_digests(
            request.images, project=self.project
        )
        if len(digests) == len(request.images):
            return tuple(
                PublishedImage(image=image, digest=digests[image])
                for image in request.images
            )

        config = _cloud_build_config(request)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="osmosis-cloud-build-"
        ) as config_file:
            json.dump(config, config_file)
            config_file.flush()
            command = [
                "gcloud",
                "builds",
                "submit",
                "--quiet",
                "--async",
                "--suppress-logs",
                "--project",
                self.project,
                "--region",
                self.region,
                "--timeout=2400",
                "--config",
                config_file.name,
                "--gcs-source-staging-dir",
                f"gs://{self.project}_cloudbuild/source",
                "--format=json",
                str(request.context),
            ]
            returncode, output, error = await _run_command(command)

        if returncode != 0:
            details = error.strip() or output.strip()
            raise RuntimeError(
                f"Google Cloud Build failed for {request.images[0]}: {details}"
            )
        payload = _cloud_build_payload(output)
        build_id = payload.get("id")
        if not isinstance(build_id, str) or not build_id:
            raise RuntimeError("Google Cloud Build did not return a build ID")

        output = await self._wait_for_build(build_id, request.images[0])
        digests = _cloud_build_digests(output)
        if len(digests) != len(request.images):
            digests.update(
                await _google_image_digests(request.images, project=self.project)
            )
        return tuple(
            PublishedImage(image=image, digest=digests.get(image))
            for image in request.images
        )

    async def _wait_for_build(self, build_id: str, image: str) -> str:
        terminal_failures = {
            "CANCELLED",
            "EXPIRED",
            "FAILURE",
            "INTERNAL_ERROR",
            "TIMEOUT",
        }
        while True:
            command = [
                "gcloud",
                "builds",
                "describe",
                build_id,
                "--project",
                self.project,
                "--region",
                self.region,
                "--format=json",
            ]
            returncode, output, error = await _run_command(command)
            if returncode != 0:
                details = error.strip() or output.strip()
                raise RuntimeError(
                    f"Could not read Google Cloud Build {build_id}: {details}"
                )
            payload = _cloud_build_payload(output)
            status = payload.get("status")
            if status == "SUCCESS":
                return output
            if status in terminal_failures:
                failure = payload.get("failureInfo")
                detail = failure.get("detail") if isinstance(failure, dict) else None
                raise RuntimeError(
                    f"Google Cloud Build failed for {image}: {detail or status}"
                )
            await asyncio.sleep(2)


def _image_repository(repository: str) -> str:
    repository = repository.strip().rstrip("/")
    if not repository:
        raise ValueError("image_repository must be non-empty")
    if "@" in repository or ":" in repository.rsplit("/", 1)[-1]:
        raise ValueError("image_repository must not include a tag or digest")
    return repository


def _harbor_image_reference(
    repository: str, *, environment_name: str, environment_id: str
) -> str:
    """Match Harbor's content-addressed GKE/SkyPilot image convention."""
    return f"{repository}/{environment_name}:{environment_id}"


def _select_builder(
    build_system: BuildSystem,
    *,
    buildx_builder: str | None,
    gcp_project: str | None,
    gcp_region: str | None,
) -> ImageBuilder:
    if build_system is BuildSystem.BUILDX:
        if gcp_project is not None or gcp_region is not None:
            raise ValueError(
                "gcp_project and gcp_region require build_system='google-cloud-build'"
            )
        return BuildxImageBuilder(builder=buildx_builder)

    if buildx_builder is not None:
        raise ValueError("buildx_builder can only be used with build_system='buildx'")
    if gcp_project is None or gcp_region is None:
        raise ValueError(
            "gcp_project and gcp_region are required with "
            "build_system='google-cloud-build'"
        )
    return GoogleCloudBuildImageBuilder(project=gcp_project, region=gcp_region)


def _task_environment(task_path: Path) -> tuple[Task, Path, Path]:
    from harbor.models.task.task import Task

    task = HarborTask(task_path)
    task.reject_symlinks()
    source = task.path / "environment"
    dockerfile = source / "Dockerfile"
    if not dockerfile.is_file():
        raise ValueError(
            f"Harbor task {task.path.name!r} has no environment/Dockerfile; "
            "prebuild currently supports Dockerfile environments only"
        )
    return Task(task.path, disable_verification=True), source, dockerfile


async def build_and_publish(
    dataset: str,
    *,
    image_repository: str,
    build_system: BuildSystem | str = BuildSystem.BUILDX,
    platform: str = "linux/amd64",
    buildx_builder: str | None = None,
    gcp_project: str | None = None,
    gcp_region: str | None = None,
) -> BuildAndPublishResult:
    """Build each distinct dataset environment and publish it to a registry."""
    try:
        selected_system = BuildSystem(build_system)
    except ValueError as exc:
        choices = ", ".join(system.value for system in BuildSystem)
        raise ValueError(
            f"unknown build_system {build_system!r}; choose {choices}"
        ) from exc
    if not platform.strip():
        raise ValueError("platform must be non-empty")

    repository = _image_repository(image_repository)
    builder = _select_builder(
        selected_system,
        buildx_builder=buildx_builder,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
    )
    resolved = await resolve_harbor_dataset(dataset, disable_verification=True)

    from harbor.environments.definition import environment_content_hash

    grouped: dict[str, tuple[Path, Path, list[str], list[str]]] = {}
    for task_path in resolved.task_paths:
        task, context, dockerfile = _task_environment(task_path)
        content_hash = environment_content_hash(
            context,
            docker_image=task.config.environment.docker_image,
        )
        image = _harbor_image_reference(
            repository,
            environment_name=task.short_name,
            environment_id=content_hash,
        )
        existing = grouped.get(content_hash)
        if existing is None:
            grouped[content_hash] = (
                context,
                dockerfile,
                [task.name],
                [image],
            )
        else:
            existing[2].append(task.name)
            if image not in existing[3]:
                existing[3].append(image)

    environments: list[PublishedEnvironment] = []
    for content_hash, (context, dockerfile, task_names, image_names) in grouped.items():
        published = await builder.build_and_push(
            BuildRequest(
                context=context,
                dockerfile=dockerfile,
                images=tuple(image_names),
                platform=platform,
            )
        )
        environments.append(
            PublishedEnvironment(
                content_hash=content_hash,
                images=published,
                task_names=tuple(task_names),
            )
        )

    return BuildAndPublishResult(
        dataset=dataset,
        dataset_path=resolved.path,
        task_count=len(resolved.task_paths),
        environments=tuple(environments),
    )


__all__ = [
    "BuildAndPublishResult",
    "BuildRequest",
    "BuildSystem",
    "BuildxImageBuilder",
    "GoogleCloudBuildImageBuilder",
    "ImageBuilder",
    "PublishedEnvironment",
    "PublishedImage",
    "build_and_publish",
]
