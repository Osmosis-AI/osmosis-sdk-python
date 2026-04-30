"""Discovery helpers for bundled project templates."""

from __future__ import annotations

from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    cookbook_root,
    list_templates,
    template_path,
)

__all__ = [
    "TemplateNotFoundError",
    "cookbook_root",
    "list_templates",
    "template_path",
]
