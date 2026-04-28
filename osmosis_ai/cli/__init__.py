"""Osmosis CLI framework: console utilities and command entry point.

The CLI entry point (main) is accessed via osmosis_ai.cli.main:main
and is NOT eagerly imported here to avoid circular imports with
modules that only need the shared infrastructure.
"""
