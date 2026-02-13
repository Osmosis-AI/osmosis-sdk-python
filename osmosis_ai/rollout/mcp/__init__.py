"""MCP tool support for git-sync users.

Provides :class:`MCPAgentLoop` — a built-in :class:`RolloutAgentLoop` that
loads tools from a FastMCP server directory so that ``osmosis eval --tools``
and ``osmosis test --tools`` work without any user-written AgentLoop code.
"""

from osmosis_ai.rollout.mcp.agent_loop import MCPAgentLoop
from osmosis_ai.rollout.mcp.loader import MCPLoadError, load_mcp_server

__all__ = ["MCPAgentLoop", "MCPLoadError", "load_mcp_server"]
