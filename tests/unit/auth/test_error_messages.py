"""Tests that verify each auth error state produces the correct user-facing message.

These are integration-style tests: they call the CLI entry point and check
stderr output, ensuring the exception type -> main.py handler -> message chain
is correct end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
)
from osmosis_ai.platform.constants import MSG_ENV_TOKEN_INVALID


class TestPlatformRequestMessages:
    """Test platform_request() error messages for each failure mode."""

    @patch(
        "osmosis_ai.platform.auth.platform_client.load_credentials", return_value=None
    )
    def test_no_credentials_does_not_say_expired(self, _mock: object) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        with pytest.raises(CLIError, match="Not logged in") as exc_info:
            platform_request("/api/test")
        assert exc_info.value.code == "AUTH_REQUIRED"


class TestMainExceptionHandlerMessages:
    """Test that main() maps exceptions to the correct stderr output."""

    @patch("osmosis_ai.cli.main._registered", True)
    def test_cli_error_shows_message_directly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from osmosis_ai.cli.main import main

        with patch(
            "osmosis_ai.cli.main.app",
            side_effect=CLIError(
                "This command requires an Osmosis workspace directory."
            ),
        ):
            code = main([])

        assert code == 1
        assert "Osmosis workspace directory" in capsys.readouterr().err

    @patch("osmosis_ai.cli.main._registered", True)
    def test_cli_error_preserves_bracketed_section_names(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from osmosis_ai.cli.main import main

        with patch(
            "osmosis_ai.cli.main.app",
            side_effect=CLIError("Missing [experiment] section in train.toml"),
        ):
            code = main([])

        assert code == 1
        assert "Missing [experiment] section" in capsys.readouterr().err

    @patch("osmosis_ai.cli.main._registered", True)
    def test_auth_expired_shows_session_expired(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from osmosis_ai.cli.main import main

        with patch(
            "osmosis_ai.cli.main.app",
            side_effect=AuthenticationExpiredError(),
        ):
            code = main([])

        assert code == 1
        captured = capsys.readouterr().err
        assert "session has expired" in captured.lower()

    @patch("osmosis_ai.cli.main._registered", True)
    def test_auth_expired_preserves_env_token_guidance(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from osmosis_ai.cli.main import main

        with patch(
            "osmosis_ai.cli.main.app",
            side_effect=AuthenticationExpiredError(MSG_ENV_TOKEN_INVALID),
        ):
            code = main([])

        assert code == 1
        captured = capsys.readouterr().err
        assert "OSMOSIS_TOKEN environment variable is invalid or expired" in captured
        assert "unset OSMOSIS_TOKEN" in captured
