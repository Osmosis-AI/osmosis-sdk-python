"""Per-invocation explicit workspace selection for platform CLI commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmosis_ai.cli._click_compat import Context


_workspace_name_var: ContextVar[str | None] = ContextVar(
    "osmosis_workspace_name",
    default=None,
)


def get_workspace_name() -> str | None:
    """Return the root ``--workspace`` selection for the active CLI invocation."""
    return _workspace_name_var.get()


def install_workspace_name(ctx: Context, workspace_name: str | None) -> None:
    """Install and reset an explicit workspace selection with the root context."""
    token: Token[str | None] = _workspace_name_var.set(workspace_name)
    ctx.call_on_close(lambda: _workspace_name_var.reset(token))


@contextmanager
def override_workspace_name(workspace_name: str | None) -> Iterator[None]:
    """Temporarily select a workspace, primarily for focused request tests."""
    token = _workspace_name_var.set(workspace_name)
    try:
        yield
    finally:
        _workspace_name_var.reset(token)


__all__ = ["get_workspace_name", "install_workspace_name", "override_workspace_name"]
