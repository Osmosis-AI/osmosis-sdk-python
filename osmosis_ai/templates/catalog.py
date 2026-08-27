"""SDK-owned workspace template catalog.

The public workspace-template repository contains user-editable starter files.
Control metadata such as recipe ownership and scaffold write allow-lists live
in the SDK so local user edits cannot change CLI behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TemplateRecipe:
    """A template recipe known by this SDK version."""

    name: str
    description: str
    files: tuple[Path, ...]
    owned_dirs: tuple[Path, ...]
    next_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScaffoldEntry:
    """A file or directory marker needed for workspace directory repair."""

    dest: str
    official: bool = False


def _path(value: str) -> Path:
    return Path(value)


def _recipe(name: str, description: str) -> TemplateRecipe:
    return TemplateRecipe(
        name=name,
        description=description,
        files=(
            _path(f"rollouts/{name}/**"),
            _path(f"configs/eval/{name}.toml"),
            _path(f"configs/training/{name}.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path(f"rollouts/{name}"),),
        next_steps=(
            f"pip install -e rollouts/{name}",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            f"osmosis eval submit configs/eval/{name}.toml",
            f"osmosis train submit configs/training/{name}.toml",
        ),
    )


TEMPLATE_RECIPES: tuple[TemplateRecipe, ...] = (
    _recipe("multiply-local-strands", "Local Strands multiply rollout"),
    _recipe("multiply-local-openai", "Local OpenAI Agents multiply rollout"),
    _recipe("multiply-harbor-strands", "Harbor-backed Strands multiply rollout"),
)


OFFICIAL_AGENT_SCAFFOLD_PATHS: tuple[Path, ...] = (
    _path("AGENTS.md"),
    _path("CLAUDE.md"),
    _path("configs/AGENTS.md"),
)

REQUIRED_WORKSPACE_DIRS: tuple[Path, ...] = (
    _path("rollouts"),
    _path("configs/training"),
    _path("configs/eval"),
    _path("data"),
)


def required_workspace_paths() -> tuple[str, ...]:
    """Return canonical required workspace directories with display slashes."""
    return tuple(f"{path.as_posix()}/" for path in REQUIRED_WORKSPACE_DIRS)


def recipes_by_name() -> dict[str, TemplateRecipe]:
    return {recipe.name: recipe for recipe in TEMPLATE_RECIPES}


def shared_template_files() -> frozenset[Path]:
    """Files explicitly shared by multiple SDK-owned template recipes."""
    return frozenset({Path("data/multiply.jsonl")})


__all__ = [
    "OFFICIAL_AGENT_SCAFFOLD_PATHS",
    "REQUIRED_WORKSPACE_DIRS",
    "ScaffoldEntry",
    "TemplateRecipe",
    "recipes_by_name",
    "required_workspace_paths",
    "shared_template_files",
]
