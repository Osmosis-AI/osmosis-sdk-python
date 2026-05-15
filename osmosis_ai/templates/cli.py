"""Business logic for ``osmosis template`` commands.

Templates are read from the SDK-owned catalog and copied into the canonical
workspace directory layout for ``eval run`` and ``train submit``.
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
from osmosis_ai.templates.catalog import shared_template_files
from osmosis_ai.templates.registry import (
    TemplateNotFoundError,
    iter_template_files,
    list_templates,
    template_recipe,
)
from osmosis_ai.templates.source import workspace_template_root


def _require_workspace_directory() -> Path:
    """Resolve the active workspace directory."""
    from osmosis_ai.platform.cli.workspace_directory_contract import (
        resolve_workspace_directory,
    )

    return resolve_workspace_directory()


def _format_unknown_template(name: str) -> CLIError:
    available = list_templates()
    if available:
        listing = ", ".join(available)
        hint = f"Available templates: {listing}."
    else:
        hint = "No templates are currently available."
    return CLIError(
        f"Template '{name}' not found. {hint}",
        code="NOT_FOUND",
    )


# ── osmosis template list ────────────────────────────────────────


def list_command() -> CommandResult | None:
    """List templates."""
    names = list_templates()

    output = get_output_context()
    if output.format is OutputFormat.rich:
        if not names:
            console.print(
                "No templates are currently available.",
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


def _same_file_contents(left: Path, right: Path) -> bool:
    try:
        return (
            left.is_file()
            and right.is_file()
            and left.read_bytes() == right.read_bytes()
        )
    except OSError:
        return False


def _ensure_within_workspace_directory(dest: Path, workspace_directory: Path) -> None:
    """Refuse writes outside the workspace directory."""
    try:
        dest.resolve().relative_to(workspace_directory)
    except ValueError as exc:
        raise CLIError(
            f"Refusing to write template file outside the workspace directory: {dest}",
            code="VALIDATION",
        ) from exc


def _format_conflicts(conflicts: list[str], name: str) -> CLIError:
    listing = "\n  ".join(sorted(set(conflicts)))
    return CLIError(
        "Refusing to overwrite existing files in the workspace directory:\n"
        f"  {listing}\n"
        f"\nRe-run with `osmosis template apply {name} --force` to replace them.",
        code="CONFLICT",
    )


def _format_blocked_owned_paths(blocked_paths: list[str]) -> CLIError:
    listing = "\n  ".join(sorted(set(blocked_paths)))
    return CLIError(
        "Refusing to replace non-directory template-owned paths:\n"
        f"  {listing}\n"
        "\nMove or replace these paths; `--force` only replaces existing directories.",
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
        f"osmosis dataset upload data/{name}.jsonl",
        f"Edit configs/training/{name}.toml with the uploaded dataset ID and target model",
        'git add . && git commit -m "add rollout template"',
        "git push",
        f"osmosis train submit configs/training/{name}.toml",
    ]


def _copy_template(
    name: str,
    workspace_directory: Path,
    *,
    force: bool = False,
) -> tuple[list[Path], list[str]]:
    """Copy template files and return destinations plus written paths."""
    try:
        recipe = template_recipe(name)
    except TemplateNotFoundError as exc:
        raise _format_unknown_template(name) from exc

    workspace_directory_resolved = workspace_directory.resolve()
    written: list[str] = []
    template_root = workspace_template_root(refresh=True)
    file_rels = iter_template_files(name, root=template_root)
    shared_file_rels = shared_template_files()
    owned_dests = [workspace_directory_resolved / rel for rel in recipe.owned_dirs]

    # Containment guard upfront.
    for rel in file_rels:
        _ensure_within_workspace_directory(
            workspace_directory_resolved / rel, workspace_directory_resolved
        )
    for owned_dest in owned_dests:
        _ensure_within_workspace_directory(owned_dest, workspace_directory_resolved)

    if not force:
        conflicts: list[str] = []
        for owned_dest in owned_dests:
            if owned_dest.exists():
                conflicts.append(
                    owned_dest.relative_to(workspace_directory_resolved).as_posix()
                    + "/"
                )
        for rel in file_rels:
            src = template_root / rel
            dest = workspace_directory_resolved / rel
            if any(_is_under(dest, owned_dest) for owned_dest in owned_dests):
                continue
            if dest.exists():
                if rel in shared_file_rels and _same_file_contents(src, dest):
                    continue
                conflicts.append(rel.as_posix())
        if conflicts:
            raise _format_conflicts(conflicts, name)

    # Reset only SDK-catalog-owned directories wholesale.
    blocked_owned_paths: list[str] = []
    for owned_dest in owned_dests:
        if owned_dest.is_symlink() or (owned_dest.exists() and not owned_dest.is_dir()):
            blocked_owned_paths.append(
                owned_dest.relative_to(workspace_directory_resolved).as_posix()
            )
    if blocked_owned_paths:
        raise _format_blocked_owned_paths(blocked_owned_paths)
    for owned_dest in owned_dests:
        if owned_dest.is_dir():
            shutil.rmtree(owned_dest)

    for rel in file_rels:
        src = template_root / rel
        dest = workspace_directory_resolved / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written.append(rel.as_posix())

    top_level: list[Path] = list(owned_dests)
    seen: set[Path] = set(top_level)
    for rel in file_rels:
        dest = workspace_directory_resolved / rel
        if any(_is_under(dest, owned_dest) for owned_dest in owned_dests):
            continue
        parent = dest.parent
        if parent not in seen:
            seen.add(parent)
            top_level.append(parent)

    return top_level, written


def apply_command(name: str, *, force: bool = False) -> CommandResult | None:
    """Apply a template into the active workspace directory layout."""
    workspace_directory = _require_workspace_directory()
    destinations, written = _copy_template(name, workspace_directory, force=force)

    rel_destinations = [
        dest.relative_to(workspace_directory).as_posix() + "/" for dest in destinations
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
