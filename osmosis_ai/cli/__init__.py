"""Osmosis CLI framework: shared errors and console utilities.

The CLI entry point (main) is accessed via osmosis_ai.cli.main:main
and is NOT eagerly imported here to avoid circular imports with
modules that only need the shared infrastructure (CLIError, Console).
"""

from .errors import CLIError

__all__ = [
    "CLIError",
]
