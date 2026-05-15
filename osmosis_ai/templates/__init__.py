"""Discovery helpers for workspace template recipes."""

from __future__ import annotations

from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    list_templates,
    template_path,
    template_recipe,
)

__all__ = [
    "TemplateNotFoundError",
    "list_templates",
    "template_path",
    "template_recipe",
]
