"""Template-backed scaffold repair for existing Osmosis projects.

This module owns the retained scaffold primitives used to repair an
already-created Osmosis project checkout. It is not a bootstrap flow: it does
not create projects, initialize git repositories, or make initial commits.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from osmosis_ai.consts import PACKAGE_VERSION

_PLUGIN_REPO_DEFAULT = "Osmosis-AI/osmosis-plugins"
_PLUGIN_MARKETPLACE_DEFAULT = "osmosis"

_TEMPLATES = files("osmosis_ai.platform.cli") / "templates"


@dataclass(frozen=True, slots=True)
class ScaffoldEntry:
    """A single file to create during scaffold repair.

    Attributes:
        template: Relative path inside the templates package directory.
            Empty string means write an empty file (used for ``.gitkeep``).
        dest: Relative path inside the target project directory.
        render: Whether to apply ``str.format_map()`` variable substitution.
        overwrite_on_update: If True, overwrite existing file during update mode.
    """

    template: str
    dest: str
    render: bool = False
    overwrite_on_update: bool = False


SCAFFOLD: list[ScaffoldEntry] = [
    ScaffoldEntry("", ".osmosis/cache/.gitkeep"),
    ScaffoldEntry("", "rollouts/.gitkeep"),
    ScaffoldEntry("", "configs/eval/.gitkeep"),
    ScaffoldEntry("", "configs/training/.gitkeep"),
    ScaffoldEntry("", "data/.gitkeep"),
    ScaffoldEntry("program.md.tpl", ".osmosis/research/program.md"),
    ScaffoldEntry("pyproject.toml.tpl", "pyproject.toml", render=True),
    ScaffoldEntry("README.md.tpl", "README.md", render=True),
    ScaffoldEntry("gitignore.tpl", ".gitignore"),
    ScaffoldEntry("configs/training/default.toml.tpl", "configs/training/default.toml"),
    ScaffoldEntry("AGENTS.md.tpl", "AGENTS.md", render=True, overwrite_on_update=True),
    ScaffoldEntry("CLAUDE.md.tpl", "CLAUDE.md", overwrite_on_update=True),
    ScaffoldEntry(
        "configs/AGENTS.md.tpl", "configs/AGENTS.md", overwrite_on_update=True
    ),
    ScaffoldEntry(
        "claude/settings.json.tpl",
        ".claude/settings.json",
        render=True,
        overwrite_on_update=True,
    ),
]

AGENT_REFRESH_PATHS = {
    "AGENTS.md",
    "CLAUDE.md",
    "configs/AGENTS.md",
    ".claude/settings.json",
}


def _plugin_repo() -> str:
    """GitHub repo (`owner/name`) hosting the Osmosis plugin marketplace."""
    return os.environ.get("OSMOSIS_PLUGIN_REPO") or _PLUGIN_REPO_DEFAULT


def _plugin_marketplace() -> str:
    """Marketplace name as declared in the plugin repo's `marketplace.json`."""
    return os.environ.get("OSMOSIS_PLUGIN_MARKETPLACE") or _PLUGIN_MARKETPLACE_DEFAULT


def _render_template(name: str, variables: dict[str, str]) -> str:
    """Read a scaffold template file and apply variable substitution."""
    text = (_TEMPLATES / name).read_text(encoding="utf-8")
    return text.format_map(variables)


def _scaffold_variables(project_name: str) -> dict[str, str]:
    return {
        "name": project_name,
        "sdk_version": PACKAGE_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "plugin_repo": _plugin_repo(),
        "plugin_marketplace": _plugin_marketplace(),
    }


def write_scaffold(target: Path, project_name: str, *, update: bool = False) -> None:
    """Write scaffold files into *target* from the SCAFFOLD manifest.

    In normal mode, files that already exist are skipped. In update mode,
    entries with ``overwrite_on_update=True`` are overwritten and all other
    existing files are left untouched.
    """
    variables = _scaffold_variables(project_name)

    for entry in SCAFFOLD:
        dest = target / entry.dest
        if dest.exists() and not (update and entry.overwrite_on_update):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not entry.template:
            dest.write_text("", encoding="utf-8")
        elif entry.render:
            dest.write_text(
                _render_template(entry.template, variables), encoding="utf-8"
            )
        else:
            dest.write_bytes((_TEMPLATES / entry.template).read_bytes())


__all__ = [
    "AGENT_REFRESH_PATHS",
    "SCAFFOLD",
    "ScaffoldEntry",
    "write_scaffold",
]
