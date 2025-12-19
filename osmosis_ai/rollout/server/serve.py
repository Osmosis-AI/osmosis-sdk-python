"""Server entry point for RolloutAgentLoop implementations.

This module provides the serve_agent_loop() function for starting
a RolloutServer with validation and configuration.

Example:
    from osmosis_ai.rollout import RolloutAgentLoop
    from osmosis_ai.rollout.server import serve_agent_loop

    class MyAgent(RolloutAgentLoop):
        name = "my_agent"
        # ...

    # Start server with validation
    serve_agent_loop(MyAgent(), port=9000)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, TYPE_CHECKING

from osmosis_ai.rollout._compat import FASTAPI_AVAILABLE, UVICORN_AVAILABLE
from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.validator import (
    AgentLoopValidationError,
    ValidationResult,
    validate_agent_loop,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.config.settings import RolloutSettings

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9000
DEFAULT_HOST = "0.0.0.0"


class ServeError(Exception):
    """Raised when server cannot be started."""

    pass


def serve_agent_loop(
    agent_loop: RolloutAgentLoop,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    validate: bool = True,
    log_level: str = "info",
    reload: bool = False,
    settings: Optional["RolloutSettings"] = None,
) -> None:
    """Start a RolloutServer for the given agent loop.

    This function validates the agent loop (optional), creates a FastAPI
    application, and starts it with uvicorn.

    Args:
        agent_loop: The RolloutAgentLoop instance to serve.
        host: Host to bind to. Defaults to "0.0.0.0".
        port: Port to bind to. Defaults to 9000.
        validate: Whether to validate the agent loop before starting.
                  Defaults to True.
        log_level: Uvicorn log level. Defaults to "info".
        reload: Whether to enable auto-reload (for development).
                Defaults to False.
        settings: Optional RolloutSettings for configuration.

    Raises:
        ImportError: If FastAPI or uvicorn is not installed.
        AgentLoopValidationError: If validation fails and validate=True.
        ServeError: If server cannot be started.

    Example:
        from osmosis_ai.rollout.server import serve_agent_loop

        serve_agent_loop(MyAgentLoop(), port=9000)

        # Skip validation (not recommended)
        serve_agent_loop(MyAgentLoop(), port=9000, validate=False)
    """
    # Check dependencies
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for serve_agent_loop(). "
            "Install it with: pip install osmosis-ai[server]"
        )

    if not UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is required for serve_agent_loop(). "
            "Install it with: pip install osmosis-ai[server]"
        )

    # Validate agent loop
    if validate:
        result = validate_agent_loop(agent_loop)
        _log_validation_result(result)
        result.raise_if_invalid()

    # Create app
    from osmosis_ai.rollout.server.app import create_app

    app = create_app(agent_loop, settings=settings)

    # Start server
    import uvicorn

    logger.info(
        "Starting RolloutServer: agent=%s, host=%s, port=%d",
        agent_loop.name,
        host,
        port,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def validate_and_report(
    agent_loop: RolloutAgentLoop,
    *,
    verbose: bool = False,
) -> ValidationResult:
    """Validate an agent loop and print a report.

    This is a convenience function for CLI usage that validates
    the agent loop and prints human-readable output.

    Args:
        agent_loop: The RolloutAgentLoop instance to validate.
        verbose: Whether to print detailed output. Defaults to False.

    Returns:
        ValidationResult with validation status.

    Example:
        result = validate_and_report(MyAgentLoop(), verbose=True)
        if not result.valid:
            sys.exit(1)
    """
    result = validate_agent_loop(agent_loop)
    _log_validation_result(result, verbose=verbose)
    return result


def _log_validation_result(result: ValidationResult, *, verbose: bool = False) -> None:
    """Log validation result to console."""
    if result.valid:
        print(f"Agent loop '{result.agent_name}' validated successfully.")
        print(f"  - Tools: {result.tool_count}")
        if result.warnings:
            print(f"  - Warnings: {len(result.warnings)}")
            if verbose:
                for warning in result.warnings:
                    print(f"    - {warning}")
    else:
        print(f"Agent loop validation failed with {len(result.errors)} error(s):", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        if result.warnings and verbose:
            print(f"\nWarnings ({len(result.warnings)}):", file=sys.stderr)
            for warning in result.warnings:
                print(f"  - {warning}", file=sys.stderr)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ServeError",
    "serve_agent_loop",
    "validate_and_report",
]
