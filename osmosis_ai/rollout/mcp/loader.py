"""MCP server loader for git-sync users.

Loads a FastMCP server instance from a user's MCP directory (containing main.py),
making all registered @mcp.tool() functions available for use in MCPAgentLoop.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
import types
from typing import Any


class MCPLoadError(Exception):
    """Error raised when loading an MCP server fails."""

    pass


def _clear_module_tree(prefix: str) -> None:
    """Remove modules under ``prefix`` from ``sys.modules``."""
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


@contextlib.contextmanager
def _temporary_sys_path(path: str):
    """Temporarily prepend ``path`` to ``sys.path`` and restore exactly."""
    original = list(sys.path)
    # Ensure local sibling imports resolve to the MCP directory during import.
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def load_mcp_server(mcp_path: str) -> Any:
    """Load a FastMCP server instance from a directory.

    The directory must contain a ``main.py`` that creates a FastMCP instance
    and registers tools via ``@mcp.tool()``.  Importing ``main.py`` triggers
    all tool registrations.

    Args:
        mcp_path: Path to the MCP directory containing ``main.py``.

    Returns:
        The FastMCP server instance found in ``main.py``.

    Raises:
        MCPLoadError: If the directory is invalid, ``main.py`` is missing,
            fastmcp is not installed, or no FastMCP instance is found.
    """
    # Validate directory
    mcp_dir = os.path.abspath(mcp_path)
    if not os.path.isdir(mcp_dir):
        raise MCPLoadError(f"MCP directory does not exist: {mcp_dir}")

    main_py = os.path.join(mcp_dir, "main.py")
    if not os.path.isfile(main_py):
        raise MCPLoadError(
            f"No main.py found in MCP directory: {mcp_dir}\n"
            "The --mcp directory must contain a main.py with a FastMCP instance."
        )

    # Check fastmcp is installed
    try:
        from fastmcp import FastMCP  # noqa: F401
    except ImportError:
        raise MCPLoadError(
            "fastmcp is not installed. Install it with: pip install osmosis-ai[mcp]"
        )

    # Import main.py as a submodule of a synthetic package:
    # - supports relative imports like `from .tools import ...`
    # - avoids leaking user module names into the global namespace
    dir_hash = hashlib.md5(mcp_dir.encode()).hexdigest()[:8]
    package_name = f"_osmosis_mcp_{dir_hash}"
    module_name = f"{package_name}.main"

    # Ensure stale modules from previous loads don't shadow filesystem changes.
    _clear_module_tree(package_name)

    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [mcp_dir]  # type: ignore[attr-defined]
    package.__file__ = os.path.join(mcp_dir, "__init__.py")
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        sys.modules.pop(package_name, None)
        raise MCPLoadError(f"Cannot create module spec for: {main_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        # Allow both relative imports (`from .x`) and legacy local absolute
        # imports (`import tools`) without persisting path changes.
        with _temporary_sys_path(mcp_dir):
            spec.loader.exec_module(module)
    except Exception as e:
        _clear_module_tree(package_name)
        raise MCPLoadError(f"Error importing {main_py}: {e}") from e

    # Find the FastMCP instance in the module
    from fastmcp import FastMCP

    mcp_server = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, FastMCP):
            mcp_server = obj
            break

    if mcp_server is None:
        raise MCPLoadError(
            f"No FastMCP instance found in {main_py}. "
            "Make sure your main.py creates a FastMCP instance, e.g.:\n"
            "  from fastmcp import FastMCP\n"
            "  mcp = FastMCP('my_server')"
        )

    return mcp_server


__all__ = ["MCPLoadError", "load_mcp_server"]
