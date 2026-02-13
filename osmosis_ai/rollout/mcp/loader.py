"""MCP server loader for git-sync users.

Loads a FastMCP server instance from a user's MCP directory (containing main.py),
making all registered @mcp.tool() functions available for use in MCPAgentLoop.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from typing import Any


class MCPLoadError(Exception):
    """Error raised when loading an MCP server fails."""

    pass


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
    # Check fastmcp is installed
    try:
        from fastmcp import FastMCP  # noqa: F401
    except ImportError:
        raise MCPLoadError(
            "fastmcp is not installed. Install it with: pip install osmosis-ai[mcp]"
        )

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

    # Add MCP directory to sys.path so relative imports work
    if mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)

    # Import main.py with a unique module name to avoid conflicts
    dir_hash = hashlib.md5(mcp_dir.encode()).hexdigest()[:8]
    module_name = f"_osmosis_mcp_{dir_hash}"

    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise MCPLoadError(f"Cannot create module spec for: {main_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up on failure
        sys.modules.pop(module_name, None)
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
