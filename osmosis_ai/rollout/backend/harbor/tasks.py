"""Harbor task directories. Host-side only — nothing here runs in a container.

A HarborTask is one task directory on disk (instruction.md, environment/,
tests/). materialize() copies it into a per-rollout run directory and stages
that rollout's files; Harbor then builds the container from the copy. Harbor's
content-addressed image cache dedupes builds across identical copies.
"""

from __future__ import annotations

import json
import shutil
from enum import StrEnum
from pathlib import Path

from osmosis_ai.rollout.container.files import INPUT_FILENAME, ContainerInput


class TaskMode(StrEnum):
    TEMPLATE = "template"
    DATASET = "dataset"


class HarborTask:
    def __init__(self, path: Path):
        self.path = path.resolve()
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
    ) -> Path:
        """Copy this task into *out_dir* and stage one rollout's files.

        With grader_wheel the generated test.sh installs the wheel first, so
        grading works even when the agent phase installed nothing (native
        Harbor agents); the container input ships in tests/ for the same reason.
        """
        task_dir = out_dir / self.path.name
        shutil.rmtree(task_dir, ignore_errors=True)
        shutil.copytree(self.path, task_dir)

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
                lines.append(
                    f"uv pip install --system /tests/{grader_wheel.name} "
                    f"|| python3 -m pip install /tests/{grader_wheel.name}"
                )
            lines.append(grader_script)
            test_sh.write_text("\n".join(lines) + "\n")
            test_sh.chmod(0o755)

        return task_dir
