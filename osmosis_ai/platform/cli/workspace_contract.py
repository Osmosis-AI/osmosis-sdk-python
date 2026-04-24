"""Workspace contract helpers for structured Osmosis workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osmosis_ai.cli.errors import CLIError

_REQUIRED_DIRS = (
    ".osmosis",
    ".osmosis/research",
    "rollouts",
    "configs",
    "configs/eval",
    "configs/training",
    "data",
)

_WORKSPACE_TOML = ".osmosis/workspace.toml"


def find_workspace_root(start: Path) -> Path | None:
    """Return the nearest ancestor that looks like an Osmosis workspace."""
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / _WORKSPACE_TOML).is_file():
            return candidate
    return None


def resolve_workspace_root(start: Path | None = None) -> Path:
    """Resolve the active Osmosis workspace root from a path or the cwd."""
    candidate = start or Path.cwd()
    workspace_root = find_workspace_root(candidate)
    if workspace_root is None:
        raise CLIError(
            "Not in an Osmosis workspace.\n"
            "  Expected to find .osmosis/workspace.toml in this directory or an ancestor."
        )
    return workspace_root


def validate_workspace_contract(workspace_root: Path) -> None:
    """Ensure the canonical workspace layout exists."""
    workspace_root = workspace_root.resolve()

    missing_paths = [
        rel_path
        for rel_path in _REQUIRED_DIRS
        if not (workspace_root / rel_path).is_dir()
    ]
    if not (workspace_root / _WORKSPACE_TOML).is_file():
        missing_paths.insert(0, _WORKSPACE_TOML)

    if not missing_paths:
        return

    formatted = "\n".join(f"  - {path}" for path in missing_paths)
    raise CLIError(
        "Workspace is missing required Osmosis paths.\n"
        f"{formatted}\n"
        "\n"
        "Re-run `osmosis init` in this workspace to restore the canonical layout."
    )


def ensure_workspace_config_path(
    config_path: Path,
    workspace_root: Path,
    *,
    config_dir: str,
    command_label: str,
) -> None:
    """Require command configs to live under the canonical workspace directory."""
    config_path = config_path.resolve()
    config_root = (workspace_root / config_dir).resolve()

    try:
        config_path.relative_to(config_root)
    except ValueError as exc:
        raise CLIError(
            f"{command_label} config must live under `{config_dir}/`.\n"
            f"  Got: {config_path}\n"
            f"  Expected under: {config_root}"
        ) from exc

    if config_path.suffix != ".toml":
        raise CLIError(
            f"{command_label} config must be a TOML file under `{config_dir}/`, got: {config_path}"
        )


def _format_backend_validation_errors(errors: list[Any]) -> str:
    return "\n".join(f"  - [{error.code}] {error.message}" for error in errors)


def validate_rollout_backend(
    *,
    workspace_root: Path,
    rollout: str,
    entrypoint: str,
    command_label: str,
    grader_module: str | None = None,
    grader_config_ref: str | None = None,
) -> None:
    """Load and validate a rollout backend against the workspace contract."""
    from osmosis_ai.eval.common.cli import _resolve_grader, load_workflow
    from osmosis_ai.rollout.validator import validate_backend

    workflow_cls, workflow_config, entrypoint_module, workflow_error = load_workflow(
        rollout=rollout,
        entrypoint=entrypoint,
        quiet=True,
        console=None,
        workspace_root=workspace_root,
    )
    if workflow_error or workflow_cls is None or entrypoint_module is None:
        raise CLIError(
            f"{command_label} preflight failed for `rollouts/{rollout}/{entrypoint}`.\n"
            f"  {workflow_error or 'Failed to load workflow.'}"
        )

    try:
        grader_cls, grader_config = _resolve_grader(
            entrypoint_module,
            explicit_grader=grader_module,
            explicit_config=grader_config_ref,
        )
    except (CLIError, ImportError, TypeError, ValueError) as exc:
        raise CLIError(
            f"{command_label} preflight failed while resolving the grader.\n  {exc}"
        ) from exc

    if grader_cls is None:
        raise CLIError(
            f"{command_label} requires a concrete `Grader` for `rollouts/{rollout}/{entrypoint}`.\n"
            "  Define a Grader in the entrypoint module or configure `[grader].module`."
        )

    validation_result = validate_backend(
        workflow_cls,
        workflow_config,
        grader_cls=grader_cls,
        grader_config=grader_config,
    )
    if validation_result.valid:
        return

    raise CLIError(
        f"{command_label} preflight failed for `rollouts/{rollout}/{entrypoint}`.\n"
        f"{_format_backend_validation_errors(validation_result.errors)}"
    )


__all__ = [
    "ensure_workspace_config_path",
    "find_workspace_root",
    "resolve_workspace_root",
    "validate_rollout_backend",
    "validate_workspace_contract",
]
