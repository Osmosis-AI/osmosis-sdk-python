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
from osmosis_ai.rollout.console import Console
from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.server.api_key import generate_api_key
from osmosis_ai.rollout.validator import (
    AgentLoopValidationError,
    ValidationResult,
    validate_agent_loop,
)

if TYPE_CHECKING:
    from osmosis_ai.rollout.config.settings import RolloutSettings
    from osmosis_ai.auth.credentials import WorkspaceCredentials

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
    skip_register: bool = False,
    api_key: Optional[str] = None,
    local_debug: bool = False,
    debug_dir: Optional[str] = None,
) -> None:
    """Start a RolloutServer for the given agent loop.

    This function validates the agent loop (optional), creates a FastAPI
    application, and starts it with uvicorn. By default, it also registers
    the server with Osmosis Platform for health monitoring.

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
        skip_register: Whether to skip registering with Osmosis Platform.
                       Defaults to False. If False, requires valid login
                       credentials.
        api_key: Optional API key for authenticating incoming requests.
                 If None, a new key is generated automatically.
                 TrainGate must send this key as:
                 - Authorization: Bearer <api_key>
                 when calling this RolloutServer.
                 and is NOT related to the `osmosis login` token.
        local_debug: Local debug mode. If True, disables API key authentication
                     and forces skip_register=True. NOT for production.
        debug_dir: Optional directory for debug logging.
                   If provided, each rollout will write detailed execution
                   traces to {debug_dir}/{rollout_id}.jsonl files.
                   Disabled by default.

    Raises:
        ImportError: If FastAPI or uvicorn is not installed.
        AgentLoopValidationError: If validation fails and validate=True.
        ServeError: If server cannot be started or not logged in.

    Example:
        from osmosis_ai.rollout.server import serve_agent_loop

        serve_agent_loop(MyAgentLoop(), port=9000)

        # Skip validation (not recommended)
        serve_agent_loop(MyAgentLoop(), port=9000, validate=False)

        # Skip platform registration (for local testing)
        serve_agent_loop(MyAgentLoop(), port=9000, skip_register=True)

        # Use a custom API key (e.g., from environment variable)
        import os
        serve_agent_loop(MyAgentLoop(), api_key=os.environ.get("MY_API_KEY"))

        # Local debug mode (no API key auth, no registration)
        serve_agent_loop(MyAgentLoop(), local_debug=True)
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

    # Local debug mode: force skip registration and disable API key auth
    if local_debug:
        skip_register = True
        if api_key is not None:
            raise ServeError(
                "local_debug=True disables API key authentication; "
                "do not provide api_key in local debug mode."
            )

    # Check login status if registration is enabled
    credentials: Optional["WorkspaceCredentials"] = None
    if not skip_register:
        from osmosis_ai.auth.credentials import get_valid_credentials

        credentials = get_valid_credentials()
        if credentials is None:
            raise ServeError(
                "Not logged in. Please run 'osmosis login' first, "
                "or use skip_register=True for local testing."
            )

    # Validate agent loop
    if validate:
        result = validate_agent_loop(agent_loop)
        _log_validation_result(result)
        result.raise_if_invalid()

    # Configure API Key auth (required by default unless local_debug=True)
    api_key_provided = api_key is not None
    if local_debug:
        logger.info("Local debug mode enabled: API key authentication disabled")
        api_key = None
        api_key_provided = False
    else:
        # Generate API Key for this server instance if not provided
        if api_key is None:
            api_key = generate_api_key()
            logger.info("Generated new API key for server authentication")
        else:
            logger.info("Using provided API key for server authentication")

    # Create debug session directory with timestamp if debug_dir is provided
    debug_session_dir: Optional[str] = None
    if debug_dir:
        import os
        import time

        timestamp = int(time.time())
        debug_session_dir = os.path.join(debug_dir, str(timestamp))
        os.makedirs(debug_session_dir, exist_ok=True)
        logger.info("Debug logging enabled: session_dir=%s", debug_session_dir)

    # Create app
    from osmosis_ai.rollout.server.app import create_app

    app = create_app(
        agent_loop,
        settings=settings,
        credentials=credentials,
        server_host=host,
        server_port=port,
        api_key=api_key,
        debug_dir=debug_session_dir,
    )

    # Start server
    import uvicorn

    logger.info(
        "Starting RolloutServer: agent=%s, host=%s, port=%d",
        agent_loop.name,
        host,
        port,
    )

    # Print server info including API key
    console = Console()
    info_lines = [f"Agent:       {agent_loop.name}", f"Address:     {host}:{port}"]

    # Show detected public IP when binding to all interfaces
    if host == "0.0.0.0":
        public_ip = _get_public_ip()
        if public_ip:
            info_lines.append(f"Public IP:   {public_ip}")
        else:
            info_lines.append("Public IP:   (could not detect)")

    if local_debug:
        info_lines.append("API Key:     (disabled - local debug)")
    else:
        if api_key_provided:
            info_lines.append("API Key:     (provided)")
        else:
            info_lines.append(f"API Key:     {api_key}")

    if skip_register:
        if local_debug:
            info_lines.append("Registration: skipped (local debug mode)")
        else:
            info_lines.append("Registration: skipped (local testing mode)")

    if debug_session_dir:
        info_lines.append(f"Debug Log:   {debug_session_dir}/")

    console.print()
    console.panel("RolloutServer", "\n".join(info_lines), style="green")
    console.print()

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


def _get_public_ip() -> Optional[str]:
    """Get detected public IP address, or None if detection fails."""
    from osmosis_ai.rollout.network import detect_public_ip, PublicIPDetectionError

    try:
        return detect_public_ip()
    except PublicIPDetectionError:
        return None


def _log_validation_result(result: ValidationResult, *, verbose: bool = False) -> None:
    """Log validation result to console."""
    console = Console()
    if result.valid:
        console.print(f"Agent loop '{result.agent_name}' validated successfully.", style="green")
        console.print(f"  - Tools: {result.tool_count}")
        if result.warnings:
            console.print(f"  - Warnings: {len(result.warnings)}", style="yellow")
            if verbose:
                for warning in result.warnings:
                    console.print(f"    - {warning}", style="yellow")
    else:
        console.print_error(f"Agent loop validation failed with {len(result.errors)} error(s):")
        for error in result.errors:
            console.print_error(f"  - {error}")
        if result.warnings and verbose:
            console.print_error(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                console.print_error(f"  - {warning}")


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ServeError",
    "serve_agent_loop",
    "validate_and_report",
]
