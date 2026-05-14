"""Discovery of workspace template recipes."""

from __future__ import annotations

from pathlib import Path

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates.catalog import TemplateRecipe, recipes_by_name
from osmosis_ai.templates.source import workspace_template_root


class TemplateNotFoundError(LookupError):
    """Unknown template name."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def _validate_template_name(name: str) -> None:
    if not name or name.startswith((".", "_")) or "/" in name or "\\" in name:
        raise TemplateNotFoundError(name)


def list_templates() -> list[str]:
    """Return sorted visible recipe names."""
    return sorted(recipes_by_name())


def template_path(name: str) -> Path:
    """Resolve a recipe name to the workspace template root."""
    _validate_template_name(name)
    if name not in recipes_by_name():
        raise TemplateNotFoundError(name)
    return workspace_template_root()


def template_recipe(name: str) -> TemplateRecipe:
    """Resolve a recipe name to its SDK catalog entry."""
    _validate_template_name(name)
    recipes = recipes_by_name()
    try:
        return recipes[name]
    except KeyError as exc:
        raise TemplateNotFoundError(name) from exc


def _expand_catalog_files(root: Path, patterns: tuple[Path, ...]) -> list[Path]:
    """Expand SDK catalog file patterns against the workspace template checkout."""
    rel_paths: set[Path] = set()
    for pattern in patterns:
        pattern_text = pattern.as_posix()
        if any(part in {"*", "**"} or "*" in part for part in pattern.parts):
            if pattern.parts[-1] == "**":
                matches = sorted((root / Path(*pattern.parts[:-1])).rglob("*"))
            else:
                matches = sorted(root.glob(pattern_text))
            file_matches = [path for path in matches if path.is_file()]
            if not file_matches:
                raise CLIError(
                    f"Workspace template pattern matched no files: {pattern_text}",
                    code="NOT_FOUND",
                )
            for match in file_matches:
                rel_paths.add(match.relative_to(root))
            continue

        candidate = root / pattern
        if not candidate.is_file():
            raise CLIError(
                f"Workspace template file does not exist: {pattern_text}",
                code="NOT_FOUND",
            )
        rel_paths.add(pattern)
    return sorted(rel_paths, key=lambda path: path.as_posix())


def iter_template_files(name: str) -> list[Path]:
    """Return relative file paths declared by the SDK recipe catalog."""
    recipe = template_recipe(name)
    return _expand_catalog_files(workspace_template_root(), recipe.files)


__all__ = [
    "TemplateNotFoundError",
    "iter_template_files",
    "list_templates",
    "template_path",
    "template_recipe",
]
