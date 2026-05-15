"""Business logic for ``osmosis rollout init``.

Reads SDK-bundled scaffold templates and stamps them into the active project's
canonical layout. The templates ship as package data alongside this module, so
``rollout init`` works offline and is unaffected by user edits to the
workspace-template repo.
"""

from __future__ import annotations

import re
import shutil
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    CommandResult,
    OperationResult,
    OutputFormat,
    get_output_context,
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_RESERVED_NAMES = frozenset({"default"})
_ROLLOUT_NAME_TOKEN = "<your-rollout>"
_SCAFFOLD_PACKAGE = "osmosis_ai.templates._scaffolds.rollout"

# Source ``.tpl`` filename → project-relative destination template. ``{name}``
# is replaced with the rollout name at render time.
_SCAFFOLD_LAYOUT: tuple[tuple[str, str], ...] = (
    ("main.py.tpl", "rollouts/{name}/main.py"),
    ("pyproject.toml.tpl", "rollouts/{name}/pyproject.toml"),
    ("README.md.tpl", "rollouts/{name}/README.md"),
    ("eval.toml.tpl", "configs/eval/{name}.toml"),
    ("training.toml.tpl", "configs/training/{name}.toml"),
)


def _validate_rollout_name(name: str) -> None:
    if not name:
        raise CLIError("Rollout name must be non-empty.", code="VALIDATION")
    if name in _RESERVED_NAMES:
        raise CLIError(
            f"'{name}' is reserved and cannot be used as a rollout name.",
            code="VALIDATION",
        )
    if not _NAME_PATTERN.match(name):
        raise CLIError(
            f"Invalid rollout name '{name}'.\n"
            "  Must match ^[a-z][a-z0-9-]*$ (lowercase letters, digits, and "
            "hyphens; must start with a letter).",
            code="VALIDATION",
        )


def _planned_destinations(project_root: Path, name: str) -> dict[str, Path]:
    """Top-level project paths owned by this scaffold.

    Used for conflict detection and cleanup. Individual files inside
    ``rollouts/<name>/`` live in ``_SCAFFOLD_LAYOUT``.
    """
    return {
        "rollout_dir": project_root / "rollouts" / name,
        "eval_config": project_root / "configs" / "eval" / f"{name}.toml",
        "training_config": project_root / "configs" / "training" / f"{name}.toml",
    }


def _format_conflicts(name: str, conflicts: list[str]) -> CLIError:
    listing = "\n  ".join(conflicts)
    return CLIError(
        "Refusing to overwrite existing rollout paths:\n"
        f"  {listing}\n"
        f"\nRe-run with `osmosis rollout init {name} --force` to replace them.",
        code="CONFLICT",
    )


def _check_conflicts(project_root: Path, name: str) -> None:
    conflicts: list[str] = []
    for dest in _planned_destinations(project_root, name).values():
        if dest.exists() or dest.is_symlink():
            suffix = "/" if dest.is_dir() and not dest.is_symlink() else ""
            conflicts.append(dest.relative_to(project_root).as_posix() + suffix)
    if conflicts:
        raise _format_conflicts(name, conflicts)


def _scaffold_root() -> Traversable:
    return files(_SCAFFOLD_PACKAGE)


def _check_scaffold_files_present(scaffold_root: Traversable) -> None:
    missing = [
        tpl_name
        for tpl_name, _ in _SCAFFOLD_LAYOUT
        if not scaffold_root.joinpath(tpl_name).is_file()
    ]
    if missing:
        listing = "\n  ".join(missing)
        raise CLIError(
            "SDK scaffold package is missing files required by "
            "`osmosis rollout init`:\n"
            f"  {listing}\n"
            "\nThis usually means the osmosis-ai wheel was built without its "
            "scaffold resources; reinstall the SDK.",
            code="NOT_FOUND",
        )


def _render_to(source: Traversable, dest: Path, name: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = source.read_text(encoding="utf-8").replace(_ROLLOUT_NAME_TOKEN, name)
    dest.write_text(rendered, encoding="utf-8")


def _reset_rollout_dir(rollout_dir: Path, name: str) -> None:
    if not rollout_dir.exists() and not rollout_dir.is_symlink():
        return
    if rollout_dir.is_symlink() or not rollout_dir.is_dir():
        raise CLIError(
            f"Refusing to replace non-directory path: rollouts/{name}\n"
            "  --force only replaces existing directories; move or delete this "
            "path manually before re-running.",
            code="CONFLICT",
        )
    shutil.rmtree(rollout_dir)


def _check_force_config_targets(
    project_root: Path, destinations: dict[str, Path]
) -> None:
    blocked: list[str] = []
    for key in ("eval_config", "training_config"):
        dest = destinations[key]
        if dest.is_symlink() or dest.is_dir() or (dest.exists() and not dest.is_file()):
            blocked.append(dest.relative_to(project_root).as_posix())

    if not blocked:
        return

    listing = "\n  ".join(blocked)
    raise CLIError(
        "Refusing to replace non-regular rollout config paths:\n"
        f"  {listing}\n"
        "\nMove or delete these paths; `--force` only replaces regular config files.",
        code="CONFLICT",
    )


def init_command(name: str, *, force: bool = False) -> CommandResult | None:
    """Scaffold ``rollouts/<name>/`` and matching eval/training configs."""
    from osmosis_ai.platform.cli.project_contract import (
        resolve_project_root_from_cwd,
        validate_project_contract,
    )

    _validate_rollout_name(name)
    project_root = resolve_project_root_from_cwd()
    validate_project_contract(project_root)

    destinations = _planned_destinations(project_root, name)
    rollout_dir = destinations["rollout_dir"]
    eval_config = destinations["eval_config"]
    training_config = destinations["training_config"]

    if not force:
        _check_conflicts(project_root, name)

    scaffold_root = _scaffold_root()
    _check_scaffold_files_present(scaffold_root)

    if force:
        _check_force_config_targets(project_root, destinations)
        _reset_rollout_dir(rollout_dir, name)

    written: list[str] = []
    for tpl_name, dest_template in _SCAFFOLD_LAYOUT:
        dest = project_root / dest_template.format(name=name)
        _render_to(scaffold_root.joinpath(tpl_name), dest, name)
        written.append(dest.relative_to(project_root).as_posix())

    next_steps = [
        f"pip install -e rollouts/{name}",
        f"osmosis eval run configs/eval/{name}.toml --limit 1",
        f"osmosis train submit configs/training/{name}.toml",
    ]
    rollout_dir_rel = rollout_dir.relative_to(project_root).as_posix()
    eval_config_rel = eval_config.relative_to(project_root).as_posix()
    training_config_rel = training_config.relative_to(project_root).as_posix()

    output = get_output_context()
    if output.format is OutputFormat.rich:
        console.print(f"Initialized rollout '{name}'.", style="green")
        console.print(f"Wrote {len(written)} file(s):", style="dim")
        for path in written:
            console.print(f"  {path}", style="dim")
        console.print()
        console.print("Next:", style="dim")
        for command in next_steps:
            console.print(f"  {command}", style="dim")
        return None

    return OperationResult(
        operation="rollout.init",
        status="success",
        resource={
            "name": name,
            "project_root": str(project_root),
            "rollout_dir": rollout_dir_rel,
            "configs": {
                "eval": eval_config_rel,
                "training": training_config_rel,
            },
            "files": written,
        },
        message=f"Initialized rollout '{name}'.",
        display_next_steps=next_steps,
    )


__all__ = ["init_command"]
