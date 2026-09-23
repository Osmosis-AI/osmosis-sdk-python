"""Build and publish content-addressed Harbor task images."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from osmosis_ai.rollout.backend.harbor.dataset import resolve_harbor_dataset
from osmosis_ai.rollout.backend.harbor.tasks import HarborTask


class BuildSystem(StrEnum):
    BUILDX = "buildx"
    GOOGLE_CLOUD_BUILD = "google-cloud-build"


@dataclass(frozen=True)
class BuildRequest:
    context: Path
    dockerfile: Path
    image_base: str
    image: str
    build_args: Mapping[str, str]
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
    image: PublishedImage
    content_hash: str
    task_names: tuple[str, ...]


@dataclass(frozen=True)
class BuildAndPublishResult:
    dataset: str
    dataset_path: Path
    task_count: int
    environments: tuple[PublishedEnvironment, ...]


class ImageBuilder(Protocol):
    async def build_and_push(self, request: BuildRequest) -> PublishedImage: ...


class BuildxImageBuilder:
    """Publish through Harbor's Buildx image builder."""

    def __init__(self, *, builder: str | None = None) -> None:
        self.builder = builder

    async def build_and_push(self, request: BuildRequest) -> PublishedImage:
        from harbor.environments.docker.utils import ensure_docker_image_built

        previous_builder = os.environ.get("BUILDX_BUILDER")
        if self.builder is not None:
            os.environ["BUILDX_BUILDER"] = self.builder
        try:
            image = await ensure_docker_image_built(
                docker_name=request.image_base,
                docker_build_context=request.context,
                dockerfile_path=request.dockerfile,
                build_args=request.build_args,
                platform=request.platform,
                push=True,
            )
        finally:
            if self.builder is not None:
                if previous_builder is None:
                    os.environ.pop("BUILDX_BUILDER", None)
                else:
                    os.environ["BUILDX_BUILDER"] = previous_builder

        if image != request.image:
            raise RuntimeError(
                f"Harbor produced unexpected image name {image!r}; "
                f"expected {request.image!r}"
            )
        return PublishedImage(image=image)


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


def _cloud_build_config(request: BuildRequest) -> dict[str, object]:
    dockerfile = request.dockerfile.relative_to(request.context).as_posix()
    args = [
        "build",
        f"--file={dockerfile}",
        *[
            f"--build-arg={key}={value}"
            for key, value in sorted(request.build_args.items())
        ],
        f"--platform={request.platform}",
        f"--tag={request.image}",
        ".",
    ]
    return {
        "steps": [{"name": "gcr.io/cloud-builders/docker", "args": args}],
        "images": [request.image],
    }


def _cloud_build_digest(output: str, image: str) -> str | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    images = payload.get("results", {}).get("images", [])
    for result in images:
        if result.get("name") == image:
            digest = result.get("digest")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                return digest
    return None


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

    async def build_and_push(self, request: BuildRequest) -> PublishedImage:
        if _google_image_lookup_command(request.image, project=self.project) is None:
            raise ValueError(
                "google-cloud-build currently requires a Google Artifact Registry "
                "or Google Container Registry image repository"
            )
        digest = await _google_image_digest(request.image, project=self.project)
        if digest is not None:
            return PublishedImage(image=request.image, digest=digest)

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
                f"Google Cloud Build failed for {request.image}: {details}"
            )
        payload = _cloud_build_payload(output)
        build_id = payload.get("id")
        if not isinstance(build_id, str) or not build_id:
            raise RuntimeError("Google Cloud Build did not return a build ID")

        output = await self._wait_for_build(build_id, request.image)
        digest = _cloud_build_digest(output, request.image)
        if digest is None:
            digest = await _google_image_digest(request.image, project=self.project)
        return PublishedImage(image=request.image, digest=digest)

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


def _image_base(repository: str) -> str:
    repository = repository.strip().rstrip("/")
    if not repository:
        raise ValueError("image_repository must be non-empty")
    if "@" in repository or ":" in repository.rsplit("/", 1)[-1]:
        raise ValueError("image_repository must not include a tag or digest")
    return f"{repository}:osmosis"


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


def _stage_environment(
    task_path: Path,
    destination: Path,
) -> tuple[Path, Path]:
    task = HarborTask(task_path)
    task.reject_symlinks()
    source = task.path / "environment"
    dockerfile = source / "Dockerfile"
    if not dockerfile.is_file():
        raise ValueError(
            f"Harbor task {task.path.name!r} has no environment/Dockerfile; "
            "prebuild currently supports Dockerfile environments only"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination, destination / "Dockerfile"


async def build_and_publish(
    dataset: str,
    *,
    image_repository: str,
    build_system: BuildSystem | str = BuildSystem.BUILDX,
    platform: str = "linux/amd64",
    build_args: Mapping[str, str] | None = None,
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

    image_base = _image_base(image_repository)
    builder = _select_builder(
        selected_system,
        buildx_builder=buildx_builder,
        gcp_project=gcp_project,
        gcp_region=gcp_region,
    )
    resolved = await resolve_harbor_dataset(dataset, disable_verification=True)

    from harbor.utils.container_cache import docker_build_context_hash

    arguments = dict(build_args or {})
    grouped: dict[str, tuple[Path, Path, list[str]]] = {}
    with tempfile.TemporaryDirectory(prefix="osmosis-harbor-prebuild-") as temp:
        staging = Path(temp)
        for index, task_path in enumerate(resolved.task_paths):
            context, dockerfile = _stage_environment(task_path, staging / str(index))
            content_hash = docker_build_context_hash(
                context=context,
                dockerfile_path=dockerfile,
                build_args=arguments,
                platform=platform,
            )
            existing = grouped.get(content_hash)
            if existing is None:
                grouped[content_hash] = (context, dockerfile, [task_path.name])
            else:
                existing[2].append(task_path.name)
                shutil.rmtree(context)

        environments: list[PublishedEnvironment] = []
        for content_hash, (context, dockerfile, task_names) in grouped.items():
            image = f"{image_base}--{content_hash}"
            published = await builder.build_and_push(
                BuildRequest(
                    context=context,
                    dockerfile=dockerfile,
                    image_base=image_base,
                    image=image,
                    build_args=arguments,
                    platform=platform,
                )
            )
            environments.append(
                PublishedEnvironment(
                    image=published,
                    content_hash=content_hash,
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
