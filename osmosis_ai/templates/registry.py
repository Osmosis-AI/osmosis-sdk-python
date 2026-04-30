"""Discovery of bundled project templates."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path


class TemplateNotFoundError(LookupError):
    """Unknown template name."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def cookbook_root() -> Traversable:
    """Return the cookbook resource root."""
    return files("osmosis_ai.templates") / "cookbook"


def list_templates() -> list[str]:
    """Return sorted visible template names."""
    root = cookbook_root()
    names: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith((".", "_")):
            continue
        names.append(entry.name)
    names.sort()
    return names


def template_path(name: str) -> Traversable:
    """Resolve a template name to its resource directory."""
    if not name or name.startswith((".", "_")) or "/" in name or "\\" in name:
        raise TemplateNotFoundError(name)

    candidate = cookbook_root() / name
    if not candidate.is_dir():
        raise TemplateNotFoundError(name)
    return candidate


def iter_template_files(name: str) -> list[Path]:
    """Return relative file paths inside a template."""
    from importlib.resources import as_file

    template_root = template_path(name)
    rel_paths: list[Path] = []
    with as_file(template_root) as concrete_root:
        for fs_path in sorted(Path(concrete_root).rglob("*")):
            if not fs_path.is_file():
                continue
            rel_paths.append(fs_path.relative_to(concrete_root))
    return rel_paths


__all__ = [
    "TemplateNotFoundError",
    "cookbook_root",
    "iter_template_files",
    "list_templates",
    "template_path",
]
