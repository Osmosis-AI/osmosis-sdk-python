"""Resolve Harbor datasets from local paths, registries, or Git repositories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError


@dataclass(frozen=True)
class ResolvedHarborDataset:
    """A dataset materialized on the local filesystem."""

    path: Path
    task_paths: tuple[Path, ...]


def dataset_config_for_source(source: str) -> Any:
    """Translate a compact dataset source into Harbor configuration."""
    from harbor.models.job.config import DatasetConfig

    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return DatasetConfig(path=candidate.resolve())
    if source.startswith((".", "/", "~")):
        raise CLIError(
            f"Harbor dataset directory does not exist: {candidate.resolve()}",
            code="NOT_FOUND",
        )
    if "://" in source or source.startswith("git@"):
        return DatasetConfig(repo=source)

    name, separator, ref = source.rpartition("@")
    if not separator:
        name, ref = source, None
    if not name:
        raise CLIError("Harbor dataset must be non-empty.", code="VALIDATION")
    if "/" in name:
        return DatasetConfig(name=name, ref=ref)
    return DatasetConfig(name=name, version=ref)


async def resolve_harbor_dataset(
    source: str,
    *,
    disable_verification: bool = False,
    config_factory: Callable[[str], Any] = dataset_config_for_source,
) -> ResolvedHarborDataset:
    """Resolve a Harbor dataset and return its local task directories."""
    try:
        from harbor.tasks.client import TaskClient
        from platformdirs import user_cache_path
    except ModuleNotFoundError as exc:
        raise CLIError(
            "Using Harbor datasets requires the Harbor dependencies. Install "
            "`osmosis-ai[harbor]`.",
            code="VALIDATION",
        ) from exc

    try:
        dataset = config_factory(source)
        if disable_verification:
            task_configs = await dataset.get_task_configs(disable_verification=True)
        else:
            task_configs = await dataset.get_task_configs()
        if not task_configs:
            raise ValueError("dataset contains no valid tasks")

        if dataset.is_local():
            assert dataset.path is not None
            root = dataset.path.expanduser().resolve()
            task_paths = tuple(config.get_local_path() for config in task_configs)
            return ResolvedHarborDataset(path=root, task_paths=task_paths)

        identity = "\n".join(
            sorted(config.model_dump_json() for config in task_configs)
        )
        dataset_dir = (
            user_cache_path("osmosis")
            / "harbor-datasets"
            / sha256(identity.encode()).hexdigest()
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result = await TaskClient().download_tasks(
            task_ids=[config.get_task_id() for config in task_configs],
            output_dir=dataset_dir,
            export=True,
        )
        if not result.paths:
            raise ValueError("dataset contains no downloadable tasks")
        return ResolvedHarborDataset(
            path=dataset_dir,
            task_paths=tuple(Path(path).resolve() for path in result.paths),
        )
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            f"Could not resolve Harbor dataset {source!r}: {exc}",
            code="VALIDATION",
        ) from exc


__all__ = [
    "ResolvedHarborDataset",
    "dataset_config_for_source",
    "resolve_harbor_dataset",
]
