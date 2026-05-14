"""Business logic for ``osmosis template`` commands.

Recipes are read from the SDK-owned catalog and copied from the public
workspace template into the canonical project layout for ``eval run`` and
``train submit``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from osmosis_ai.cli.console import console
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import (
    CommandResult,
    ListColumn,
    ListResult,
    OperationResult,
    OutputFormat,
    get_output_context,
)
from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    iter_template_files,
    list_templates,
    template_recipe,
)
from osmosis_ai.templates.source import workspace_template_root


def _require_project_root() -> Path:
    """Resolve the active project root."""
    from osmosis_ai.platform.cli.project_contract import resolve_project_root

    return resolve_project_root()


def _format_unknown_template(name: str) -> CLIError:
    available = list_templates()
    if available:
        listing = ", ".join(available)
        hint = f"Available templates: {listing}."
    else:
        hint = "No recipes are currently available from the workspace template."
    return CLIError(
        f"Template '{name}' not found. {hint}",
        code="NOT_FOUND",
    )


# ── osmosis template list ────────────────────────────────────────


def list_command() -> CommandResult | None:
    """List workspace template recipes."""
    names = list_templates()

    output = get_output_context()
    if output.format is OutputFormat.rich:
        if not names:
            console.print(
                "No recipes are currently available from the workspace template.",
                style="dim",
            )
            return None
        for name in names:
            console.print(name)
        return None

    return ListResult(
        title="Templates",
        items=[{"name": name} for name in names],
        total_count=len(names),
        has_more=False,
        next_offset=None,
        columns=[ListColumn(key="name", label="Name")],
    )


# ── osmosis template apply <name> ────────────────────────────────


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _ensure_within_project(dest: Path, project_root: Path) -> None:
    """Refuse writes outside the project root."""
    try:
        dest.resolve().relative_to(project_root)
    except ValueError as exc:
        raise CLIError(
            f"Refusing to write template file outside the project root: {dest}",
            code="VALIDATION",
        ) from exc


def _format_conflicts(conflicts: list[str], name: str) -> CLIError:
    listing = "\n  ".join(sorted(set(conflicts)))
    return CLIError(
        "Refusing to overwrite existing files in the project:\n"
        f"  {listing}\n"
        f"\nRe-run with `osmosis template apply {name} --force` to replace them.",
        code="CONFLICT",
    )


def _next_steps(name: str) -> list[str]:
    """Commands to run after applying a template."""
    try:
        recipe = template_recipe(name)
    except TemplateNotFoundError:
        recipe = None
    if recipe is not None and recipe.next_steps:
        return list(recipe.next_steps)
    return [
        f"pip install -e rollouts/{name}",
        f"osmosis eval run configs/eval/{name}.toml --limit 1",
        "git push",
        "Confirm Git Sync is connected in the Osmosis Platform",
        f"osmosis train submit configs/training/{name}.toml",
    ]


def _copy_template(
    name: str,
    project_root: Path,
    *,
    force: bool = False,
) -> tuple[list[Path], list[str]]:
    """Copy template files and return destinations plus written paths."""
    try:
        recipe = template_recipe(name)
    except TemplateNotFoundError as exc:
        raise _format_unknown_template(name) from exc

    project_root_resolved = project_root.resolve()
    written: list[str] = []
    template_root = workspace_template_root()
    file_rels = iter_template_files(name)
    owned_dests = [project_root_resolved / rel for rel in recipe.owned_dirs]

    # Containment guard upfront.
    for rel in file_rels:
        _ensure_within_project(project_root_resolved / rel, project_root_resolved)
    for owned_dest in owned_dests:
        _ensure_within_project(owned_dest, project_root_resolved)

    if not force:
        conflicts: list[str] = []
        for owned_dest in owned_dests:
            if owned_dest.exists():
                conflicts.append(
                    owned_dest.relative_to(project_root_resolved).as_posix() + "/"
                )
        for rel in file_rels:
            dest = project_root_resolved / rel
            if any(_is_under(dest, owned_dest) for owned_dest in owned_dests):
                continue
            if dest.exists():
                conflicts.append(rel.as_posix())
        if conflicts:
            raise _format_conflicts(conflicts, name)

    # Reset only SDK-catalog-owned directories wholesale.
    for owned_dest in owned_dests:
        if owned_dest.exists():
            shutil.rmtree(owned_dest)

    for rel in file_rels:
        src = template_root / rel
        dest = project_root_resolved / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written.append(rel.as_posix())

    top_level: list[Path] = list(owned_dests)
    seen: set[Path] = set(top_level)
    for rel in file_rels:
        dest = project_root_resolved / rel
        if any(_is_under(dest, owned_dest) for owned_dest in owned_dests):
            continue
        parent = dest.parent
        if parent not in seen:
            seen.add(parent)
            top_level.append(parent)

    return top_level, written


def apply_command(name: str, *, force: bool = False) -> CommandResult | None:
    """Apply a workspace template recipe into the active project layout."""
    project_root = _require_project_root()
    destinations, written = _copy_template(name, project_root, force=force)

    rel_destinations = [
        dest.relative_to(project_root).as_posix() + "/" for dest in destinations
    ]

    output = get_output_context()
    next_steps = _next_steps(name)
    if output.format is OutputFormat.rich:
        console.print(f"Applied template '{name}'.", style="green")
        if rel_destinations:
            console.print("Destinations:", style="dim")
            for path in rel_destinations:
                console.print(f"  ./{path}", style="dim")
        if written:
            console.print(f"Wrote {len(written)} file(s):", style="dim")
            for path in written:
                console.print(f"  {path}", style="dim")
        console.print()
        console.print("Next:", style="dim")
        for command in next_steps:
            console.print(f"  {command}", style="dim")
        return None

    return OperationResult(
        operation="template.apply",
        status="success",
        resource={
            "name": name,
            "destinations": [str(dest) for dest in destinations],
            "files": written,
        },
        message=(
            f"Applied template '{name}' into {len(destinations)} location(s)."
            if destinations
            else f"Applied template '{name}'."
        ),
        display_next_steps=next_steps,
    )


__all__ = [
    "apply_command",
    "list_command",
]
