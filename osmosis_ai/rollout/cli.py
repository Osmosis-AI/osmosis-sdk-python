"""CLI commands for Osmosis rollout SDK.

This module provides CLI command handlers for the rollout subsystem,
including the 'serve' command.

Example:
    osmosis serve --module my_agent:agent_loop --port 9000
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional

from osmosis_ai.rollout.core.base import RolloutAgentLoop
from osmosis_ai.rollout.server.serve import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    serve_agent_loop,
    validate_and_report,
)
from osmosis_ai.rollout.validator import validate_agent_loop


class CLIError(Exception):
    """CLI-specific error."""

    pass


def _load_agent_loop(module_path: str) -> RolloutAgentLoop:
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
    import os
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


class ServeCommand:
    """Handler for `osmosis serve`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for serve command."""
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
            dest="module",
            required=True,
            help=(
                "Module path to the agent loop in format 'module:attribute'. "
                "Example: 'my_agent:agent_loop' or 'mypackage.agents:MyAgentClass'"
            ),
        )

        parser.add_argument(
            "-p",
            "--port",
            dest="port",
            type=int,
            default=DEFAULT_PORT,
            help=f"Port to bind to (default: {DEFAULT_PORT})",
        )

        parser.add_argument(
            "-H",
            "--host",
            dest="host",
            default=DEFAULT_HOST,
            help=f"Host to bind to (default: {DEFAULT_HOST})",
        )

        parser.add_argument(
            "--no-validate",
            dest="no_validate",
            action="store_true",
            default=False,
            help="Skip agent loop validation before starting",
        )

        parser.add_argument(
            "--reload",
            dest="reload",
            action="store_true",
            default=False,
            help="Enable auto-reload for development",
        )

        parser.add_argument(
            "--log-level",
            dest="log_level",
            default="info",
            choices=["debug", "info", "warning", "error", "critical"],
            help="Uvicorn log level (default: info)",
        )

        parser.add_argument(
            "--skip-register",
            dest="skip_register",
            action="store_true",
            default=False,
            help="Skip registering with Osmosis Platform (for local testing)",
        )

        parser.add_argument(
            "--local",
            "--local-debug",
            dest="local_debug",
            action="store_true",
            default=False,
            help=(
                "Local debug mode: disable API key authentication and skip registering "
                "with Osmosis Platform (NOT for production)"
            ),
        )

        parser.add_argument(
            "--api-key",
            dest="api_key",
            default=None,
            help=(
                "API key used by TrainGate to authenticate when calling this RolloutServer "
                "(sent as 'Authorization: Bearer <api_key>'). "
                "If not provided, one is generated. (NOT related to `osmosis login` token.)"
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        """Run the serve command."""
        module_path = args.module
        port = args.port
        host = args.host
        validate = not args.no_validate
        reload = args.reload
        log_level = args.log_level
        skip_register = args.skip_register
        api_key = args.api_key
        local_debug = args.local_debug

        # Load agent loop
        try:
            agent_loop = _load_agent_loop(module_path)
        except CLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Serve
        try:
            serve_agent_loop(
                agent_loop,
                host=host,
                port=port,
                validate=validate,
                log_level=log_level,
                reload=reload,
                skip_register=skip_register,
                api_key=api_key,
                local_debug=local_debug,
            )
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        return 0


class ValidateCommand:
    """Handler for `osmosis validate`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for validate command."""
        parser.set_defaults(handler=self.run)

        parser.add_argument(
            "-m",
            "--module",
            dest="module",
            required=True,
            help=(
                "Module path to the agent loop in format 'module:attribute'. "
                "Example: 'my_agent:agent_loop' or 'mypackage.agents:MyAgentClass'"
            ),
        )

        parser.add_argument(
            "-v",
            "--verbose",
            dest="verbose",
            action="store_true",
            default=False,
            help="Show detailed validation output including warnings",
        )

    def run(self, args: argparse.Namespace) -> int:
        """Run the validate command."""
        module_path = args.module
        verbose = args.verbose

        # Load agent loop
        try:
            agent_loop = _load_agent_loop(module_path)
        except CLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Validate
        result = validate_and_report(agent_loop, verbose=verbose)

        return 0 if result.valid else 1


__all__ = [
    "CLIError",
    "ServeCommand",
    "ValidateCommand",
]
