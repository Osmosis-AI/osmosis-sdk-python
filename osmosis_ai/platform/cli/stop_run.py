"""Shared confirm + POST helper for train/eval/benchmark stop commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from osmosis_ai.cli.output import OperationResult, get_output_context
from osmosis_ai.cli.prompts import require_confirmation
from osmosis_ai.platform.cli.workspace_directory_context import (
    PlatformWorkspaceContext,
    workspace_result_context,
)


def stop_run(
    *,
    noun: str,
    operation: str,
    confirm_name: str,
    yes: bool,
    context: PlatformWorkspaceContext,
    stop: Callable[[], Any],
    status_message: str,
    extra: dict[str, Any] | None = None,
) -> OperationResult:
    """Confirm, invoke *stop*, and return a success ``OperationResult``."""
    require_confirmation(
        f'Stop {noun} "{confirm_name}"?',
        yes=yes,
        default=False,
        summary=[("Name", confirm_name)],
    )
    with get_output_context().status(status_message):
        stop()
    resource: dict[str, Any] = {"name": confirm_name}
    if extra:
        resource.update(extra)
    resource.update(workspace_result_context(context))
    return OperationResult(
        operation=operation,
        status="success",
        resource=resource,
        message=f'{noun[0].upper() + noun[1:]} "{confirm_name}" stopped.',
    )
