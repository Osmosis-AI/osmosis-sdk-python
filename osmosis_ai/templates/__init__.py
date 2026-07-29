"""Discovery helpers for workspace template recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from osmosis_ai._imports import public_module_dir, resolve_lazy_export

if TYPE_CHECKING:
    from osmosis_ai.templates.registry import (
        TemplateNotFoundError,
        list_templates,
        template_recipe,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "TemplateNotFoundError": (
        "osmosis_ai.templates.registry",
        "TemplateNotFoundError",
    ),
    "list_templates": ("osmosis_ai.templates.registry", "list_templates"),
    "template_recipe": ("osmosis_ai.templates.registry", "template_recipe"),
}


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        namespace=globals(),
        exports=_EXPORTS,
    )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "TemplateNotFoundError",
    "list_templates",
    "template_recipe",
]
