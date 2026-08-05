"""Harbor task directories. Host-side only — nothing here runs in a container.

A HarborTask is one task directory on disk (instruction.md, environment/,
tests/). materialize() copies it into a per-rollout run directory and stages
that rollout's files; Harbor then builds the container from the copy. Harbor's
content-addressed image cache dedupes builds across identical copies.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

from harbor.models.task.id import GitTaskId, LocalTaskId, PackageTaskId

from osmosis_ai.rollout.container.files import INPUT_FILENAME, ContainerInput

logger: logging.Logger = logging.getLogger(__name__)

# Harness dependencies install into this venv so the task's python is never touched.
SDK_UV = "/opt/osmosis/uv"
SDK_VENV = "/opt/osmosis/venv"
SDK_REQUIREMENTS_FILENAME = "osmosis-requirements.txt"


class TaskMode(StrEnum):
    TEMPLATE = "template"
    DATASET = "dataset"


def parse_task_ref(
    ref: str, metadata: dict[str, Any]
) -> GitTaskId | LocalTaskId | PackageTaskId:
    """metadata["harbor_task"]: a local path, git checkout, or package ref."""
    if git_url := metadata.get("git_url"):
        if not metadata.get("git_commit_id"):
            logger.warning(
                "git task %r is unpinned; set metadata['git_commit_id'] to an "
                "immutable commit sha",
                ref,
            )
        return GitTaskId(
            git_url=git_url,
            git_commit_id=metadata.get("git_commit_id"),
            path=Path(ref),
        )
    if ref.startswith((".", "/", "~")):
        return LocalTaskId(path=Path(ref))
    name, _, version = ref.partition("@")
    org, slash, task = name.partition("/")
    if not slash:
        raise ValueError(
            f"harbor_task {ref!r} must be a local path (./, /, ~), a package "
            "'org/name[@ref]', or a git checkout (set metadata['git_url'])"
        )
    if (version or "latest") == "latest":
        logger.warning(
            "package task %r uses the mutable ref 'latest'; pin a sha256 digest", ref
        )
    return PackageTaskId(org=org, name=task, ref=version or "latest")


def venv_or_fallback_install(wheel: str) -> str:
    """Shell command installing *wheel* into the SDK venv when the image has
    one, else into the system python."""
    return (
        f"if [ -x {SDK_VENV}/bin/python ]; then "
        f"{SDK_UV} pip install --python {SDK_VENV}/bin/python --no-deps {wheel}; "
        f"else uv pip install --system {wheel} || python3 -m pip install {wheel}; fi"
    )


def venv_or_fallback_script(script: str) -> str:
    """Shell command running *script* from the SDK venv when present."""
    return (
        f"if [ -x {SDK_VENV}/bin/{script} ]; then {SDK_VENV}/bin/{script}; "
        f"else {script}; fi"
    )


def patch_dockerfile_with_sdk(env_dir: Path, requirements: list[str]) -> None:
    """Pre-install *requirements* into an isolated venv in the task's image.

    Appends to the final stage: a static uv binary, the requirements file, and
    a venv at /opt/osmosis with its own managed python — the task's own
    packages and runtime user are left untouched (USER root is scoped to the
    install and the stage's original USER is restored).
    """
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise ValueError(f"cannot patch Dockerfile: none found in {env_dir}")
    lines = dockerfile.read_text().splitlines()
    stage_start = max(
        (i for i, line in enumerate(lines) if re.match(r"\s*FROM\s", line, re.I)),
        default=0,
    )
    original_user = next(
        (
            line.strip()
            for line in reversed(lines[stage_start:])
            if re.match(r"\s*USER\s", line, re.I)
        ),
        None,
    )

    (env_dir / SDK_REQUIREMENTS_FILENAME).write_text("\n".join(requirements) + "\n")
    ignore = env_dir / ".dockerignore"
    if ignore.is_file():
        ignore.write_text(
            ignore.read_text().rstrip() + f"\n!{SDK_REQUIREMENTS_FILENAME}\n"
        )

    block = [
        "",
        "USER root",
        f"COPY --from=ghcr.io/astral-sh/uv:latest /uv {SDK_UV}",
        f"COPY {SDK_REQUIREMENTS_FILENAME} /opt/osmosis/requirements.txt",
        f"RUN {SDK_UV} venv {SDK_VENV} --python 3.12 && "
        f"{SDK_UV} pip install --python {SDK_VENV}/bin/python "
        "-r /opt/osmosis/requirements.txt",
    ]
    if original_user:
        block.append(original_user)
    dockerfile.write_text("\n".join([*lines, *block]) + "\n")


class HarborTask:
    def __init__(self, path: Path):
        self.path: Path = path.resolve()
        if not self.path.is_dir():
            raise ValueError(f"harbor task directory not found: {self.path}")

    @classmethod
    def from_dataset(cls, root: Path, task_id: str) -> HarborTask:
        """Look up a task under *root* by id, rejecting path escapes."""
        root = root.resolve()
        path = (root / task_id).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError(f"unknown harbor task id: {task_id!r}")
        return cls(path)

    def materialize(
        self,
        out_dir: Path,
        container_input: ContainerInput,
        grader_script: str | None = None,
        grader_wheel: Path | None = None,
        sdk_requirements: list[str] | None = None,
    ) -> Path:
        """Copy this task into *out_dir* and stage one rollout's files.

        With grader_wheel the generated test.sh installs the wheel first, so
        grading works even when the agent phase installed nothing (native
        Harbor agents); the container input ships in tests/ for the same reason.
        With sdk_requirements the copied Dockerfile pre-installs them into an
        isolated venv, so per-trial installs stop downloading dependencies.
        """
        task_dir = out_dir / self.path.name
        shutil.rmtree(task_dir, ignore_errors=True)
        shutil.copytree(self.path, task_dir)
        if sdk_requirements:
            patch_dockerfile_with_sdk(task_dir / "environment", sdk_requirements)

        if container_input.prompt:
            (task_dir / "instruction.md").write_text(
                json.dumps(container_input.prompt, default=str)
            )
        elif not (task_dir / "instruction.md").exists():
            raise ValueError(f"task {self.path.name} has no instruction and no prompt")

        container_input.write(task_dir / INPUT_FILENAME)

        test_sh = task_dir / "tests" / "test.sh"
        if grader_script and not test_sh.exists():
            test_sh.parent.mkdir(parents=True, exist_ok=True)
            lines = ["#!/bin/bash", "set -e"]
            if grader_wheel is not None:
                shutil.copy2(grader_wheel, test_sh.parent / grader_wheel.name)
                container_input.write(test_sh.parent / INPUT_FILENAME)
                lines.append(venv_or_fallback_install(f"/tests/{grader_wheel.name}"))
            lines.append(venv_or_fallback_script(grader_script))
            test_sh.write_text("\n".join(lines) + "\n")
            test_sh.chmod(0o755)

        return task_dir
