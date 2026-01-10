"""CLI commands for Osmosis rollout SDK.

This module provides CLI command handlers for the rollout subsystem,
including the 'serve' command.

Example:
    osmosis serve --module my_agent:agent_loop --port 9000
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from osmosis_ai.rollout.cli_utils import CLIError, load_agent_loop
from osmosis_ai.rollout.server.serve import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    serve_agent_loop,
    validate_and_report,
)
from osmosis_ai.rollout.validator import validate_agent_loop


# Re-export CLIError for backwards compatibility
# (load_agent_loop is private, renamed from _load_agent_loop)
_load_agent_loop = load_agent_loop


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

        parser.add_argument(
            "--log",
            dest="debug_dir",
            default=None,
            metavar="DIR",
            help=(
                "Enable logging and write execution traces to DIR. "
                "Each rollout will create a {rollout_id}.jsonl file with "
                "detailed event logs (pre-LLM state, responses, tool results, etc.)."
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
        debug_dir = args.debug_dir

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
                debug_dir=debug_dir,
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


class TestCommand:
    """Handler for `osmosis test` (delegates to test_mode.cli).

    This class acts as a proxy to avoid circular imports and keep the main CLI
    module lightweight. The actual implementation lives in test_mode.cli.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded implementation."""
        self._impl: Optional["_TestCommandImpl"] = None

    def _get_impl(self) -> "_TestCommandImpl":
        """Lazily load the actual TestCommand implementation."""
        if self._impl is None:
            from osmosis_ai.rollout.test_mode.cli import TestCommand as _TestCommandImpl

            self._impl = _TestCommandImpl()
        return self._impl

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for test command."""
        self._get_impl().configure_parser(parser)


__all__ = [
    "CLIError",
    "ServeCommand",
    "TestCommand",
    "ValidateCommand",
]
