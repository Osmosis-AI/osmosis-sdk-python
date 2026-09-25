"""Portable source-image manifests and resolved task selections.

Published manifests reference image keys. Build results contain resolved image
records per task; the two shapes intentionally have separate types even though
both use the source-v1 wire version. No transport or workspace state is needed.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from osmosis_ai.harbor_images import TaskSource, relative_path


def task_id(value: str) -> str:
    if relative_path(value) == ".":
        raise ValueError("Task ID must identify a task beneath the source root")
    return value


class SourceImageRecord(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)
    key: str = Field(min_length=1)
    image: str

    @field_validator("image")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", value):
            raise ValueError("Source image must be pinned by digest")
        return value


class SourceTaskImages(BaseModel):
    task: str
    environments: dict[str, str]

    @field_validator("task")
    @classmethod
    def _task(cls, value: str) -> str:
        return task_id(value)


class _SourceImages(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
    schema_version: Literal["source-v1"]
    hash_policy: Literal["harbor-v1"]
    source: TaskSource
    images: list[SourceImageRecord]
    named_images: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _inventory(self) -> Self:
        if len({image.key for image in self.images}) != len(self.images):
            raise ValueError("Duplicate source image keys")
        if any(
            value not in {image.image for image in self.images}
            for value in self.named_images.values()
        ):
            raise ValueError("Named image is absent from the image inventory")
        return self


class SourceImageManifest(_SourceImages):
    """Published source manifest: task environments reference image keys."""

    tasks: list[SourceTaskImages]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        if len({task.task for task in self.tasks}) != len(self.tasks):
            raise ValueError("Duplicate source task IDs")
        images = {image.key for image in self.images}
        for task in self.tasks:
            if (
                "environment" not in task.environments
                or not set(task.environments.values()) <= images
            ):
                raise ValueError("Task is missing a required image binding")
        return self

    def resolved(self) -> SourceImageBuildResult:
        images = {image.key: image.model_dump() for image in self.images}
        value: dict[str, Any] = self.model_dump()
        value["tasks"] = {
            task.task: {role: images[key] for role, key in task.environments.items()}
            for task in self.tasks
        }
        return SourceImageBuildResult.model_validate(value)


class SourceImageBuildResult(_SourceImages):
    """Resolved result: each task environment carries its immutable image record."""

    tasks: dict[str, dict[str, SourceImageRecord]]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        images = {image.key: image for image in self.images}
        for name, bindings in self.tasks.items():
            task_id(name)
            if "environment" not in bindings or any(
                images.get(image.key) != image for image in bindings.values()
            ):
                raise ValueError("Resolved task image differs from its inventory")
        return self

    def select_task_ids(self, requested: list[str] | None = None) -> list[str]:
        selected = sorted(self.tasks if requested is None else set(requested))
        if not selected or any(name not in self.tasks for name in selected):
            raise ValueError("Select task IDs from the source image result")
        return selected
