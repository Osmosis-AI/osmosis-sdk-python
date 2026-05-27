"""Shared building blocks for ``osmosis train submit`` and ``osmosis eval submit``."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from osmosis_ai.cli.errors import CLIError


class BackendValidatedParamSection(BaseModel):
    """Base for TOML sections whose value-level schema is owned by the backend.

    The SDK only enforces structure (rejects unknown keys via ``extra='forbid'``)
    and leaves value-level validation to the platform/backend. This keeps the
    SDK from having to track backend schema evolution.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


def validate_workspace_rollout_paths(
    *,
    rollout: str,
    entrypoint: str,
    workspace_directory: Path,
    command_label: str,
) -> None:
    """Validate ``rollout`` is a single-segment logical name and that
    ``<rollouts>/<rollout>/<entrypoint>`` resolves under ``rollouts/<rollout>/``.

    ``command_label`` is the human-facing prefix in error messages
    (e.g. ``"Training"`` or ``"Eval"``).
    """
    if Path(rollout).is_absolute():
        raise CLIError(f"{command_label} rollout must be a logical rollout name.")

    if rollout in ("", ".", ".."):
        raise CLIError(
            f"{command_label} rollout {rollout!r} is not a valid rollout name."
        )

    if "/" in rollout or "\\" in rollout:
        raise CLIError(
            f"{command_label} rollout {rollout!r} must be a single-segment name "
            "(no path separators)."
        )

    rollouts_root = (workspace_directory / "rollouts").resolve()
    rollout_root = (workspace_directory / "rollouts" / rollout).resolve()
    try:
        rollout_root.relative_to(rollouts_root)
    except ValueError as exc:
        raise CLIError(
            f"{command_label} rollout must resolve under the current workspace "
            "directory's rollouts directory."
        ) from exc

    entrypoint_path = (rollout_root / entrypoint).resolve()
    try:
        entrypoint_path.relative_to(rollout_root)
    except ValueError as exc:
        raise CLIError(
            f"{command_label} entrypoint must resolve under rollouts/<rollout>/ "
            "within the current workspace directory."
        ) from exc


__all__ = [
    "BackendValidatedParamSection",
    "validate_workspace_rollout_paths",
]
