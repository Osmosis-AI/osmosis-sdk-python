"""Shared CLI utilities for Osmosis rollout SDK.

This module contains common utilities used by multiple CLI modules to avoid code
duplication (DRY principle).
"""

from __future__ import annotations

import importlib
import os
import sys

from osmosis_ai.rollout.core.base import RolloutAgentLoop


class CLIError(Exception):
    """CLI-specific error."""

    pass


def load_agent_loop(module_path: str) -> RolloutAgentLoop:
    """Load an agent loop from a module path.

    Args:
        module_path: Path in format "module.path:attribute_name"
                     e.g., "my_agent:agent_loop" or "mypackage.agents:MyAgent"

    Returns:
        RolloutAgentLoop instance.

    Raises:
        CLIError: If the module or attribute cannot be loaded.
    """
    if ":" not in module_path:
        raise CLIError(
            f"Invalid module path '{module_path}'. "
            "Expected format: 'module.path:attribute_name' "
            "(e.g., 'my_agent:agent_loop' or 'mypackage.agents:MyAgent')"
        )

    module_name, attr_name = module_path.rsplit(":", 1)

    # Add current directory to sys.path if not already there
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise CLIError(f"Cannot import module '{module_name}': {e}")

    try:
        agent_loop = getattr(module, attr_name)
    except AttributeError:
        raise CLIError(
            f"Module '{module_name}' has no attribute '{attr_name}'. "
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    # If it's a class, instantiate it
    if isinstance(agent_loop, type):
        if not issubclass(agent_loop, RolloutAgentLoop):
            raise CLIError(
                f"'{attr_name}' is a class but not a RolloutAgentLoop subclass"
            )
        try:
            agent_loop = agent_loop()
        except Exception as e:
            raise CLIError(f"Cannot instantiate '{attr_name}': {e}")

    # Validate it's a RolloutAgentLoop instance
    if not isinstance(agent_loop, RolloutAgentLoop):
        raise CLIError(
            f"'{attr_name}' must be a RolloutAgentLoop instance or subclass, "
            f"got {type(agent_loop).__name__}"
        )

    return agent_loop


__all__ = [
    "CLIError",
    "load_agent_loop",
]
