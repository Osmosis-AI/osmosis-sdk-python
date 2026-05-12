"""Business logic for ``osmosis template`` commands.

Templates are local cookbook recipes copied directly into the canonical
project layout for ``eval run`` and ``train submit``.
"""

from __future__ import annotations

import shutil
from importlib.resources import as_file
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
    list_templates,
    template_path,
)


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
        hint = "No templates are currently bundled with this SDK release."
    return CLIError(
        f"Template '{name}' not found. {hint}",
        code="NOT_FOUND",
    )


# ── osmosis template list ────────────────────────────────────────


def list_command() -> CommandResult | None:
    """List bundled cookbook templates."""
    names = list_templates()

    output = get_output_context()
    if output.format is OutputFormat.rich:
        if not names:
            console.print(
                "No templates are bundled with this SDK release.",
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


def _iter_template_files(concrete_root: Path) -> list[tuple[Path, Path]]:
    """List files under a concrete template root."""
    pairs: list[tuple[Path, Path]] = []
    for src_path in sorted(concrete_root.rglob("*")):
        if not src_path.is_file():
            continue
        pairs.append((src_path, src_path.relative_to(concrete_root)))
    return pairs


def _rollout_dest_dirs(concrete_root: Path, project_root: Path) -> list[Path]:
    """List rollout directories this template writes."""
    rollouts_src = concrete_root / "rollouts"
    if not rollouts_src.is_dir():
        return []
    dests: list[Path] = []
    for rollout_dir in sorted(rollouts_src.iterdir()):
        if rollout_dir.is_dir():
            dests.append(project_root / "rollouts" / rollout_dir.name)
    return dests


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
        src_resource = template_path(name)
    except TemplateNotFoundError as exc:
        raise _format_unknown_template(name) from exc

    project_root_resolved = project_root.resolve()
    written: list[str] = []

    with as_file(src_resource) as concrete_src:
        concrete_root = Path(concrete_src)
        file_pairs = _iter_template_files(concrete_root)
        rollout_dests = _rollout_dest_dirs(concrete_root, project_root_resolved)

        # Containment guard upfront.
        for _src, rel in file_pairs:
            _ensure_within_project(project_root_resolved / rel, project_root_resolved)
        for rollout_dest in rollout_dests:
            _ensure_within_project(rollout_dest, project_root_resolved)

        if not force:
            conflicts: list[str] = []
            for rollout_dest in rollout_dests:
                if rollout_dest.exists():
                    conflicts.append(
                        rollout_dest.relative_to(project_root_resolved).as_posix() + "/"
                    )
            for _src, rel in file_pairs:
                # Files inside a rollout dir are already covered by the
                # whole-directory conflict above; surface only "loose" files
                # (configs/, top-level metadata) so the message stays focused.
                if rel.parts and rel.parts[0] == "rollouts":
                    continue
                if (project_root_resolved / rel).exists():
                    conflicts.append(rel.as_posix())
            if conflicts:
                raise _format_conflicts(conflicts, name)

        # Reset rollout dirs wholesale: the template owns these directories.
        for rollout_dest in rollout_dests:
            if rollout_dest.exists():
                shutil.rmtree(rollout_dest)

        # Copy every template file into the project root.
        for src_path, rel in file_pairs:
            dest = project_root_resolved / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            written.append(rel.as_posix())

        # Top-level destinations: rollout dirs + the unique parent dirs of
        # any non-rollout files (typically ``configs/training``,
        # ``configs/eval``). Used for human-readable output only.
        top_level: list[Path] = list(rollout_dests)
        seen: set[Path] = set(top_level)
        for _src, rel in file_pairs:
            if rel.parts and rel.parts[0] == "rollouts":
                continue
            parent = (project_root_resolved / rel).parent
            if parent not in seen:
                seen.add(parent)
                top_level.append(parent)

    return top_level, written


def apply_command(name: str, *, force: bool = False) -> CommandResult | None:
    """Apply a cookbook template into the active project's canonical layout."""
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
