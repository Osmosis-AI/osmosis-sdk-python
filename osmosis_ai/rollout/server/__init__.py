"""FastAPI server components for Osmosis rollout SDK.

This module provides the server-side components for hosting
RolloutAgentLoop implementations.

Example:
    from osmosis_ai.rollout.server import create_app, serve_agent_loop
    from my_agent import MyAgentLoop

    # Option 1: Create app manually
    app = create_app(MyAgentLoop())
    # Run with: uvicorn main:app --port 9000

    # Option 2: Use serve_agent_loop (validates and starts server)
    serve_agent_loop(MyAgentLoop(), port=9000)
"""

from osmosis_ai.rollout.server.api_key import generate_api_key, validate_api_key
from osmosis_ai.rollout.server.app import create_app
from osmosis_ai.rollout.server.state import AppState
from osmosis_ai.rollout.server.serve import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ServeError,
    serve_agent_loop,
    validate_and_report,
)

__all__ = [
    "create_app",
    "AppState",
    "serve_agent_loop",
    "validate_and_report",
    "ServeError",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "generate_api_key",
    "validate_api_key",
]
