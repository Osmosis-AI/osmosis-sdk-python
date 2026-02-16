"""Integration tests for CLI commands.

Tests CLI commands using click.testing.CliRunner with real file system
interactions and minimal mocking.
"""

from __future__ import annotations

import pytest


class TestCLICommands:
    """Test CLI commands with real invocation."""

    @pytest.mark.integration
    def test_whoami_unauthenticated(self):
        """osmosis whoami shows appropriate message when not logged in."""
        pytest.skip("TODO: implement whoami integration test")

    @pytest.mark.integration
    def test_validate_command(self):
        """osmosis validate checks agent module correctly."""
        pytest.skip("TODO: implement validate integration test")
