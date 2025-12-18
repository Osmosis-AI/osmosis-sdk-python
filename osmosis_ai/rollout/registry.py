"""Registry for RolloutAgentLoop implementations."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, TYPE_CHECKING

from osmosis_ai.rollout.core.exceptions import AgentLoopNotFoundError

if TYPE_CHECKING:
    from osmosis_ai.rollout.core.base import RolloutAgentLoop


class AgentLoopRegistry:
    """Registry for RolloutAgentLoop implementations.

    Provides a central place to register and discover agent loop implementations.
    All operations are thread-safe.

    Example:
        registry = AgentLoopRegistry()
        registry.register(MyAgentLoop())
        registry.register(AnotherAgentLoop())

        loop = registry.get("my_agent")
        if loop:
            app = create_app(loop)
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._loops: Dict[str, "RolloutAgentLoop"] = {}
        self._lock = threading.Lock()

    def register(self, loop: "RolloutAgentLoop") -> None:
        """Register an agent loop implementation.

        This method is thread-safe.

        Args:
            loop: The agent loop instance to register.

        Raises:
            ValueError: If an agent loop with the same name is already registered.
        """
        with self._lock:
            if loop.name in self._loops:
                raise ValueError(
                    f"Agent loop '{loop.name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )
            self._loops[loop.name] = loop

    def unregister(self, name: str) -> bool:
        """Unregister an agent loop by name.

        This method is thread-safe.

        Args:
            name: The name of the agent loop to unregister.

        Returns:
            True if the agent loop was unregistered, False if not found.
        """
        with self._lock:
            if name in self._loops:
                del self._loops[name]
                return True
            return False

    def get(self, name: str) -> Optional["RolloutAgentLoop"]:
        """Get an agent loop by name.

        This method is thread-safe.

        Args:
            name: The name of the agent loop.

        Returns:
            The agent loop instance, or None if not found.
        """
        with self._lock:
            return self._loops.get(name)

    def list_names(self) -> List[str]:
        """List all registered agent loop names.

        This method is thread-safe.

        Returns:
            List of registered agent loop names.
        """
        with self._lock:
            return list(self._loops.keys())

    def clear(self) -> None:
        """Clear all registered agent loops.

        This method is thread-safe.
        """
        with self._lock:
            self._loops.clear()


# Global registry instance
_REGISTRY = AgentLoopRegistry()


def register_agent_loop(loop: "RolloutAgentLoop") -> None:
    """Register an agent loop with the global registry.

    Args:
        loop: The agent loop instance to register.

    Raises:
        ValueError: If an agent loop with the same name is already registered.

    Example:
        class MyAgentLoop(RolloutAgentLoop):
            name = "my_agent"
            ...

        register_agent_loop(MyAgentLoop())
    """
    _REGISTRY.register(loop)


def unregister_agent_loop(name: str) -> bool:
    """Unregister an agent loop from the global registry.

    Args:
        name: The name of the agent loop to unregister.

    Returns:
        True if the agent loop was unregistered, False if not found.
    """
    return _REGISTRY.unregister(name)


def get_agent_loop(name: str) -> "RolloutAgentLoop":
    """Get an agent loop from the global registry.

    Args:
        name: The name of the agent loop.

    Returns:
        The agent loop instance.

    Raises:
        AgentLoopNotFoundError: If no agent loop with the given name is registered.

    Example:
        loop = get_agent_loop("my_agent")
        app = create_app(loop)
    """
    loop = _REGISTRY.get(name)
    if loop is None:
        raise AgentLoopNotFoundError(name, _REGISTRY.list_names())
    return loop


def list_agent_loops() -> List[str]:
    """List all registered agent loop names in the global registry.

    Returns:
        List of registered agent loop names.
    """
    return _REGISTRY.list_names()
