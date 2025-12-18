"""FastAPI server components for Osmosis rollout SDK.

This module provides the server-side components for hosting
RolloutAgentLoop implementations.

Example:
    from osmosis_ai.rollout.server import create_app
    from my_agent import MyAgentLoop

    app = create_app(MyAgentLoop())
    # Run with: uvicorn main:app --port 9000
"""

from osmosis_ai.rollout.server.app import create_app
from osmosis_ai.rollout.server.state import AppState

__all__ = [
    "create_app",
    "AppState",
]
