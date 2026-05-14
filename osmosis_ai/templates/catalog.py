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
    """A file or directory marker needed for project repair."""

    dest: str
    content: str = ""
    official: bool = False


def _path(value: str) -> Path:
    return Path(value)


TEMPLATE_RECIPES: tuple[TemplateRecipe, ...] = (
    TemplateRecipe(
        name="multiply-local-strands",
        description="Local Strands multiply rollout",
        files=(
            _path("rollouts/multiply-local-strands/**"),
            _path("configs/eval/multiply-local-strands.toml"),
            _path("configs/training/multiply-local-strands.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-local-strands"),),
        next_steps=(
            "pip install -e rollouts/multiply-local-strands",
            "osmosis eval run configs/eval/multiply-local-strands.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-local-strands.toml",
        ),
    ),
    TemplateRecipe(
        name="multiply-local-openai",
        description="Local OpenAI Agents multiply rollout",
        files=(
            _path("rollouts/multiply-local-openai/**"),
            _path("configs/eval/multiply-local-openai.toml"),
            _path("configs/training/multiply-local-openai.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-local-openai"),),
        next_steps=(
            "pip install -e rollouts/multiply-local-openai",
            "osmosis eval run configs/eval/multiply-local-openai.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-local-openai.toml",
        ),
    ),
    TemplateRecipe(
        name="multiply-harbor-strands",
        description="Harbor-backed Strands multiply rollout",
        files=(
            _path("rollouts/multiply-harbor-strands/**"),
            _path("configs/eval/multiply-harbor-strands.toml"),
            _path("configs/training/multiply-harbor-strands.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-harbor-strands"),),
        next_steps=(
            "pip install -e rollouts/multiply-harbor-strands",
            "osmosis eval run configs/eval/multiply-harbor-strands.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-harbor-strands.toml",
        ),
    ),
)


OFFICIAL_AGENT_SCAFFOLD_PATHS: tuple[Path, ...] = (
    _path("AGENTS.md"),
    _path("CLAUDE.md"),
    _path("configs/AGENTS.md"),
    _path(".claude/settings.json"),
)

REQUIRED_SCAFFOLD_DIRS: tuple[Path, ...] = (
    _path(".osmosis/cache"),
    _path("rollouts"),
    _path("configs/eval"),
    _path("configs/training"),
    _path("data"),
)


def recipes_by_name() -> dict[str, TemplateRecipe]:
    return {recipe.name: recipe for recipe in TEMPLATE_RECIPES}


def official_agent_scaffold_paths() -> tuple[Path, ...]:
    return OFFICIAL_AGENT_SCAFFOLD_PATHS


__all__ = [
    "OFFICIAL_AGENT_SCAFFOLD_PATHS",
    "REQUIRED_SCAFFOLD_DIRS",
    "ScaffoldEntry",
    "TemplateRecipe",
    "official_agent_scaffold_paths",
    "recipes_by_name",
]
