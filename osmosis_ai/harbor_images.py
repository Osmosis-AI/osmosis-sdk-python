"""Versioned image identities shared by Harbor builders and rollout gateways.

The v1 policy is Linux/amd64, Git file modes, and Harbor's container-context
hash. A full SHA-256 disambiguates Harbor's short hash. Publication is an
immutable snapshot: networked Docker builds need not be reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

POLICY = "harbor-v1"
PLATFORM = "linux/amd64"
MAX_SOURCE_BYTES = 2 * 1024**3


def relative_path(value: str) -> str:
    if value == ".":
        return value
    if (
        not value
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("Task path must be relative to the repository root")
    return value


def github_repository(value: str) -> str:
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9_.-]+)/?",
        value,
    )
    if not match:
        raise ValueError("Task source must be an HTTPS GitHub repository URL")
    owner, repo = match.groups()
    repo = repo.removesuffix(".git")
    if repo in ("", ".", ".."):
        raise ValueError("Invalid GitHub repository")
    return f"https://github.com/{owner.lower()}/{repo.lower()}"


@dataclass(frozen=True)
class TaskSource:
    repository: str
    path: str
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", github_repository(self.repository))
        object.__setattr__(self, "revision", self.revision.lower())
        relative_path(self.path)
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Task source revision must be a full Git commit SHA")

    def repository_id(self, organization_id: str) -> str:
        """GAR IDs preserve source identity and avoid collisions across workspaces."""
        if not re.fullmatch(r"[a-f0-9-]{36}", organization_id):
            raise ValueError("Registry namespace requires a workspace UUID")
        name = self.repository.removeprefix("https://github.com/") + "/" + self.path
        identity = json.dumps([organization_id, name], separators=(",", ":"))
        suffix = hashlib.sha256(identity.encode()).hexdigest()[:24]
        slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:35]
        return f"ht-{slug}-{suffix}"


def fetch_source(source: TaskSource, token: str, destination: Path) -> None:
    """Fetch identical GitHub archives in builders and gateways.

    Git checkouts can apply attributes or filters that change bytes. Both sides
    use this pinned archive path so those transformations cannot change hashes.
    """
    import httpx

    if not token:
        raise ValueError("GitHub installation credential is required")
    if destination.exists():
        raise ValueError("Task source destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    api = source.repository.replace(
        "https://github.com/", "https://api.github.com/repos/"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    with tempfile.TemporaryDirectory(
        prefix="harbor-git-", dir=destination.parent
    ) as directory:
        root = Path(directory)
        archive = root / "source.tar.gz"
        try:
            with (
                httpx.Client(timeout=120, follow_redirects=False) as client,
                ExitStack() as stack,
            ):
                response = stack.enter_context(
                    client.stream(
                        "GET", f"{api}/tarball/{source.revision}", headers=headers
                    )
                )
                if response.status_code == 302:
                    location = response.headers.get("location", "")
                    target = None
                    try:
                        target = urlsplit(location)
                        valid = (
                            target.scheme == "https"
                            and target.hostname == "codeload.github.com"
                            and not target.username
                            and not target.password
                            and target.port in (None, 443)
                        )
                    except ValueError:
                        valid = False
                    if not valid or target is None:
                        raise ValueError(
                            "GitHub returned an unexpected archive location"
                        )
                    response.close()
                    response = stack.enter_context(
                        # The installation header also authenticates codeload.
                        # Keep GitHub's signed query out of HTTP request logs.
                        client.stream(
                            "GET",
                            target._replace(query="", fragment="").geturl(),
                            headers=headers,
                        )
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError("GitHub source is temporarily unavailable")
                if response.status_code != 200:
                    raise ValueError(
                        f"GitHub source download failed (HTTP {response.status_code})"
                    )
                size = 0
                with archive.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_SOURCE_BYTES:
                            raise ValueError("GitHub source archive exceeds 2 GiB")
                        output.write(chunk)
        except httpx.HTTPError:
            raise RuntimeError(
                "Could not download the pinned GitHub task source"
            ) from None
        checkout = root / "checkout"
        with tarfile.open(archive) as contents:
            size = 0
            for index, member in enumerate(contents):
                size += member.size
                if index >= 100_000 or size > MAX_SOURCE_BYTES:
                    raise ValueError("GitHub source exceeds extraction limits")
                contents.extract(member, checkout, filter="data")
        entries = list(checkout.iterdir())
        if len(entries) != 1 or not entries[0].is_dir() or entries[0].is_symlink():
            raise ValueError("GitHub source must contain one root directory")
        entries[0].rename(destination)


@contextmanager
def normalized_context(context: Path) -> Iterator[Path]:
    """Never change source files; normalize only the effective build context."""
    context = context.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="harbor-context-") as directory:
        root = Path(directory) / "context"
        for path in context.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(context):
                raise ValueError("Environment symlink escapes its build context")
            if not (path.is_symlink() or path.is_file() or path.is_dir()):
                raise ValueError("Unsupported build context file type")
        shutil.copytree(context, root, symlinks=True)
        for path in [root, *root.rglob("*")]:
            if not path.is_symlink():
                path.chmod(
                    0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
                )
        yield root


@dataclass(frozen=True)
class EnvironmentIdentity:
    environment_hash: str
    context_sha256: str
    policy: str = POLICY

    @property
    def tag(self) -> str:
        return f"{self.policy}-{self.environment_hash}-{self.context_sha256}"


def environment_identity(
    context: Path,
    *,
    source_image: str | None = None,
    build_args: Mapping[str, str] | None = None,
) -> EnvironmentIdentity:
    """Hash a normalized context (use normalized_context before calling)."""
    from harbor.utils.container_cache import docker_build_context_hash

    if version("harbor") != "0.22.0":
        raise ValueError("harbor-v1 source images require harbor==0.22.0")

    args = dict(build_args or {})
    if source_image:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]+", source_image):
            raise ValueError("Invalid prebuilt image reference")
        payload = json.dumps(
            [POLICY, PLATFORM, source_image, args], sort_keys=True
        ).encode()
        return EnvironmentIdentity(
            hashlib.blake2b(payload, digest_size=8).hexdigest(),
            hashlib.sha256(payload).hexdigest(),
        )
    if not (context / "Dockerfile").is_file():
        raise ValueError("Harbor environment needs a Dockerfile or docker_image")
    entries = []
    for root, dirs, files in context.walk(follow_symlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = root / name
            mode = path.lstat().st_mode
            content = (
                os.readlink(path).encode() if stat.S_ISLNK(mode) else path.read_bytes()
            )
            entries.append(
                [
                    path.relative_to(context).as_posix(),
                    mode,
                    hashlib.sha256(content).hexdigest(),
                ]
            )
    payload = json.dumps(
        [POLICY, PLATFORM, args, entries], sort_keys=True, separators=(",", ":")
    )
    return EnvironmentIdentity(
        docker_build_context_hash(context=context, build_args=args, platform=PLATFORM),
        hashlib.sha256(payload.encode()).hexdigest(),
    )


def task_environments(directory: Path) -> dict[str, tuple[Path, Any]]:
    from harbor.models.task.config import TaskConfig
    from harbor.models.task.paths import TaskPaths
    from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config

    paths = TaskPaths(directory)
    config = TaskConfig.model_validate(tomllib.loads(paths.config_path.read_text()))
    result = {"environment": (paths.environment_dir, config.environment)}
    for index, step in enumerate(config.steps or [None]):
        verifier = resolve_effective_verifier_env_config(config, step)
        if verifier is None:
            continue
        target = (
            f"steps.{index}.verifier.environment" if step else "verifier.environment"
        )
        explicit = (
            step and step.verifier.environment is not None
        ) or config.verifier.environment is not None
        context = paths.step_tests_dir(step.name) if step else paths.tests_dir
        if not context.exists():
            context = paths.tests_dir
        result[target] = (context if explicit else paths.environment_dir, verifier)
    for context, environment in result.values():
        if environment.os.value != "linux":
            raise ValueError("Source images support Linux environments")
        if any(
            (context / name).exists()
            for name in (
                "docker-compose.yaml",
                "docker-compose.yml",
                "compose.yaml",
                "compose.yml",
            )
        ):
            raise ValueError("Multi-service Harbor environments are not supported")
    return result


def task_identities(directory: Path) -> dict[str, EnvironmentIdentity]:
    result = {}
    for role, (context, environment) in task_environments(directory).items():
        if environment.docker_image:
            result[role] = environment_identity(
                context, source_image=environment.docker_image
            )
        else:
            with normalized_context(context) as normalized:
                result[role] = environment_identity(normalized)
    return result


def bind_task_images(directory: Path, bindings: Mapping[str, str]) -> None:
    """Bind only an ephemeral trial copy, preserving inherited verifier settings."""
    import toml
    from harbor.models.task.config import TaskConfig

    environments = task_environments(directory)
    if set(bindings) != set(environments):
        raise ValueError("Image bindings do not cover every task environment")
    path = directory / "task.toml"
    raw = tomllib.loads(path.read_text())
    for role, image in bindings.items():
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Resolved image must be pinned by digest")
        current: Any = raw
        for part in role.split("."):
            current = (
                current[int(part)]
                if isinstance(current, list)
                else current.setdefault(part, {})
            )
        current.update(environments[role][1].model_dump(mode="json", exclude_none=True))
        current["docker_image"] = image
    TaskConfig.model_validate(raw)
    path.write_text(toml.dumps(raw))


async def materialize_source_tasks(
    repository: Path, path: str, destination: Path, *, allow_empty: bool = False
) -> list[str]:
    """Discover Git folders or resolve a Harbor Hub dataset.toml's pinned tasks."""
    source = repository / relative_path(path)
    if (repository / ".gitmodules").exists():
        raise ValueError("Git submodules are unsupported; commit task files directly")
    if not source.resolve(strict=True).is_relative_to(repository.resolve()):
        raise ValueError("Task source escapes repository")
    manifest_path = source if source.is_file() else source / "dataset.toml"
    destination.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        from harbor.models.dataset.manifest import DatasetManifest
        from harbor.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
        from harbor.tasks.client import TaskClient

        manifest = DatasetManifest.from_toml_file(manifest_path)
        names = [task.name for task in manifest.tasks]
        if len(set(names)) != len(names) or not names:
            raise ValueError("Dataset must contain uniquely named tasks")
        ids: list[GitTaskId | LocalTaskId | PackageTaskId] = [
            PackageTaskId(org=task.org, name=task.short_name, ref=task.digest)
            for task in manifest.tasks
        ]
        downloaded = await TaskClient().download_tasks(
            ids, output_dir=destination / ".download"
        )
        for name, downloaded_path in zip(names, downloaded.paths, strict=True):
            validate_task_files(downloaded_path)
            shutil.copytree(downloaded_path, destination / name, symlinks=True)
        shutil.rmtree(destination / ".download", ignore_errors=True)
        return names
    names = []
    for root, dirs, files in source.walk(follow_symlinks=False):
        dirs[:] = sorted(name for name in dirs if name != ".git")
        if any((root / name).is_symlink() for name in dirs + files):
            raise ValueError("Task sources must not contain symlinks")
        if "task.toml" in files:
            name = root.relative_to(source).as_posix()
            if name == ".":
                name = Path(path).name if path != "." else "task"
            names.append(name)
            validate_task_files(root)
            shutil.copytree(root, destination / name)
            dirs.clear()
    if not names and not allow_empty:
        raise ValueError("Task source contains no Harbor tasks")
    return names


def validate_task_files(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(
                "Harbor task sources must contain only regular files and directories"
            )
        if path.is_file():
            with path.open("rb") as file:
                if file.read(80).startswith(
                    b"version https://git-lfs.github.com/spec/v1\n"
                ):
                    raise ValueError(
                        "Git LFS task files are unsupported; put large assets in the task image"
                    )
