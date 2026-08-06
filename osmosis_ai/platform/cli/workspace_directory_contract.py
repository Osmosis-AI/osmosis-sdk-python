"""Workspace directory contract helpers for structured Osmosis directories.

A workspace directory is the local on-disk repository linked to a Platform
workspace, distinct from the remote tenant managed by the platform.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.templates.catalog import required_workspace_paths


def _start_dir(start: Path) -> Path:
    current = start.resolve()
    return current.parent if current.is_file() else current


def find_workspace_directory(start: Path) -> Path | None:
    """Return the Git worktree top-level containing start, if any."""
    from osmosis_ai.platform.cli.workspace_repo import git_worktree_top_level

    return git_worktree_top_level(_start_dir(start))


def resolve_workspace_directory(start: Path | None = None) -> Path:
    """Resolve the active Osmosis workspace directory from a path or the cwd."""
    workspace_directory = find_workspace_directory(start or Path.cwd())
    if workspace_directory is None:
        raise CLIError(
            "Run this command from an Osmosis workspace directory created by Platform.",
            code="WORKSPACE_REQUIRED",
        )
    return workspace_directory.resolve()


def resolve_workspace_directory_from_cwd(cwd: Path | None = None) -> Path:
    """Resolve the active Osmosis workspace directory from cwd's Git worktree."""
    return resolve_workspace_directory(cwd or Path.cwd())


def missing_workspace_directory_paths(workspace_directory: Path) -> list[str]:
    workspace_directory = workspace_directory.resolve()
    return [
        rel_path
        for rel_path in required_workspace_paths()
        if not (workspace_directory / rel_path).is_dir()
    ]


def validate_workspace_directory_contract(workspace_directory: Path) -> None:
    """Ensure the workspace directory contains the required Osmosis scaffold paths."""
    missing_paths = missing_workspace_directory_paths(workspace_directory)
    if not missing_paths:
        return

    formatted = "\n".join(f"  - {path}" for path in missing_paths)
    raise CLIError(
        "This workspace directory is missing required Osmosis scaffold paths.\n"
        f"{formatted}\n"
        "\n"
        "Run `osmosis doctor --fix` in this Git repository to restore the scaffold."
    )


def ensure_context_path(
    path: Path,
    workspace_directory: Path,
    *,
    required_dir: str,
    label: str,
    suffix: str | None = None,
) -> Path:
    """Resolve a context-bearing path relative to workspace directory and require containment."""
    required_path = Path(required_dir)
    if required_path.is_absolute() or ".." in required_path.parts:
        raise CLIError(
            f"required_dir must be relative and must not contain '..': {required_dir}"
        )

    candidate = path if path.is_absolute() else workspace_directory / path
    resolved = candidate.resolve()
    required_root = (workspace_directory / required_path).resolve()
    try:
        resolved.relative_to(required_root)
    except ValueError as exc:
        raise CLIError(
            f"{label} must live under `{required_dir}/`.\n"
            f"  Got: {resolved}\n"
            f"  Expected under: {required_root}"
        ) from exc
    if suffix is not None and resolved.suffix != suffix:
        raise CLIError(
            f"{label} must be a {suffix} file under `{required_dir}/`, got: {resolved}"
        )
    return resolved


def ensure_workspace_directory_config_path(
    config_path: Path,
    workspace_directory: Path,
    *,
    config_dir: str,
    command_label: str,
) -> None:
    """Require command configs to live under the canonical workspace directory."""
    ensure_context_path(
        config_path,
        workspace_directory,
        required_dir=config_dir,
        label=f"{command_label} config",
        suffix=".toml",
    )


def _unsatisfied_rollout_requirements(rollout_dir: Path) -> list[str]:
    """Declared requirements this environment does not satisfy.

    Preflight imports the rollout into the workspace-root environment, not the
    rollout's own, so the two can diverge. Only declared specifiers are checked
    against installed versions; extras and transitive resolution are left to the
    resolver.
    """
    pyproject = rollout_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with open(pyproject, "rb") as f:
            declared = tomllib.load(f).get("project", {}).get("dependencies")
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        # A malformed pyproject.toml is the resolver's error to report.
        return []
    if not isinstance(declared, list):
        return []

    unsatisfied: list[str] = []
    for raw in declared:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if requirement.url:
            # A direct URL or VCS pin carries no version to compare against.
            continue
        try:
            have = installed_version(requirement.name)
        except PackageNotFoundError:
            unsatisfied.append(f"{requirement.name} is not installed")
            continue
        if requirement.specifier and not requirement.specifier.contains(
            have, prereleases=True
        ):
            unsatisfied.append(
                f"{requirement.name} {have} does not satisfy {requirement.specifier}"
            )
    return unsatisfied


def validate_rollout_backend(
    *,
    workspace_directory: Path,
    rollout: str,
    entrypoint: str,
    command_label: str,
) -> list[str]:
    """Load a rollout entrypoint and let its backend validate itself.

    Returns warnings for checks that could not run. Raises :class:`CLIError`
    only when the rollout is genuinely invalid.
    """
    from osmosis_ai.platform.cli.rollout_entrypoint import load_rollout_entrypoint
    from osmosis_ai.platform.cli.shared_config import validate_workspace_rollout_paths

    # The path check below allows `<name>/..`, which points at `rollouts/`
    # itself and loads the wrong pyproject.toml.
    validate_workspace_rollout_paths(
        rollout=rollout,
        entrypoint=entrypoint,
        workspace_directory=workspace_directory,
        command_label=command_label,
    )

    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_dir = (rollouts_root / rollout).resolve()
    if rollout_dir.is_relative_to(rollouts_root):
        unsatisfied = _unsatisfied_rollout_requirements(rollout_dir)
        if unsatisfied:
            return [
                f"Skipped the `rollouts/{rollout}` backend preflight: this environment "
                f"does not satisfy the rollout's declared dependencies "
                f"({'; '.join(unsatisfied)}). The server validates it after installing them."
            ]

    # Importing the entrypoint constructs module-level backends and servers;
    # misconfigurations surface as import-time errors. The CLI validates
    # nothing itself and never scans the namespace for classes.
    try:
        load_rollout_entrypoint(rollout_dir, entrypoint)
    except ModuleNotFoundError as exc:
        # An undeclared dependency, which the gate above cannot see.
        return [
            f"Skipped the `rollouts/{rollout}` backend preflight: {exc}. "
            "The server validates it after installing the rollout's dependencies."
        ]
    except Exception as exc:
        detail = str(exc)
        if not isinstance(exc, (CLIError, ImportError, TypeError, ValueError)):
            detail = f"{type(exc).__name__}: {detail}"
        raise CLIError(
            f"{command_label} preflight failed for `rollouts/{rollout}/{entrypoint}`.\n"
            f"  {detail}"
        ) from exc

    return []


__all__ = [
    "ensure_context_path",
    "ensure_workspace_directory_config_path",
    "find_workspace_directory",
    "missing_workspace_directory_paths",
    "resolve_workspace_directory",
    "resolve_workspace_directory_from_cwd",
    "validate_rollout_backend",
    "validate_workspace_directory_contract",
]
