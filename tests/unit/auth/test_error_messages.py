"""Tests that verify each auth error state produces the correct user-facing message.

These are integration-style tests: they call the CLI entry point and check
stderr output, ensuring the exception type -> main.py handler -> message chain
is correct end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)


def _make_credentials(*, expired: bool = False) -> Credentials:
    now = datetime.now(UTC)
    offset = timedelta(days=-1) if expired else timedelta(days=30)
    return Credentials(
        access_token="test-token",
        token_type="Bearer",
        expires_at=now + offset,
        created_at=now,
        user=UserInfo(id="u1", email="test@test.com", name="Test"),
        token_id=None,
    )


class TestRequireCredentialsMessages:
    """Test require_credentials() raises the right exception for each state."""

    @patch("osmosis_ai.platform.cli.utils.load_credentials", return_value=None)
    def test_no_credentials_raises_cli_error_not_logged_in(self, _mock: object) -> None:
        from osmosis_ai.platform.cli.utils import require_credentials

        with pytest.raises(CLIError, match="Not logged in"):
            require_credentials()

    @patch("osmosis_ai.platform.cli.utils.load_credentials")
    def test_expired_credentials_raises_auth_expired(self, mock_load: object) -> None:
        from osmosis_ai.platform.cli.utils import require_credentials

        mock_load.return_value = _make_credentials(expired=True)

        with pytest.raises(AuthenticationExpiredError):
            require_credentials()

    @patch("osmosis_ai.platform.cli.utils.load_credentials")
    def test_valid_credentials_returns_them(self, mock_load: object) -> None:
        from osmosis_ai.platform.cli.utils import require_credentials

        creds = _make_credentials()
        mock_load.return_value = creds

        result = require_credentials()
        assert result is creds


class TestGetActiveWorkspaceMessages:
    """Test _get_active_workspace_name() error message."""

    @patch(
        "osmosis_ai.platform.cli.utils.get_active_workspace_name",
        return_value=None,
    )
    def test_no_workspace_raises_correct_message(self, _mock: object) -> None:
        from osmosis_ai.platform.cli.utils import _get_active_workspace_name

        with pytest.raises(CLIError, match="No workspace selected"):
            _get_active_workspace_name()

    @patch(
        "osmosis_ai.platform.cli.utils.get_active_workspace_name",
        return_value=None,
    )
    def test_no_workspace_does_not_say_not_logged_in(self, _mock: object) -> None:
        from osmosis_ai.platform.cli.utils import _get_active_workspace_name

        with pytest.raises(CLIError) as exc_info:
            _get_active_workspace_name()

        assert "Not logged in" not in str(exc_info.value)
        assert "login" not in str(exc_info.value).lower()


class TestRequireAuthMessages:
    """Test _require_auth() checks credentials before workspace."""

    @patch("osmosis_ai.platform.cli.utils.load_credentials", return_value=None)
    @patch(
        "osmosis_ai.platform.cli.utils.get_active_workspace_name",
        return_value=None,
    )
    def test_no_login_says_not_logged_in_not_no_workspace(
        self, _ws_mock: object, _cred_mock: object
    ) -> None:
        """When both credentials and workspace are missing, error should say 'Not logged in'."""
        from osmosis_ai.platform.cli.utils import _require_auth

        with pytest.raises(CLIError, match="Not logged in"):
            _require_auth()

    @patch("osmosis_ai.platform.cli.utils.load_credentials")
    @patch(
        "osmosis_ai.platform.cli.utils.get_active_workspace_name",
        return_value=None,
    )
    def test_logged_in_but_no_workspace_says_no_workspace(
        self, _ws_mock: object, mock_load: object
    ) -> None:
        """When logged in but no workspace, error should say 'No workspace selected'."""
        from osmosis_ai.platform.cli.utils import _require_auth

        mock_load.return_value = _make_credentials()

        with pytest.raises(CLIError, match="No workspace selected"):
            _require_auth()


class TestPlatformRequestMessages:
    """Test platform_request() error messages for each failure mode."""

    @patch(
        "osmosis_ai.platform.auth.platform_client.load_credentials", return_value=None
    )
    def test_no_credentials_does_not_say_expired(self, _mock: object) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        with pytest.raises(CLIError, match="Not logged in"):
            platform_request("/api/test")

    @patch(
        "osmosis_ai.platform.auth.platform_client.get_active_workspace_id",
        return_value=None,
    )
    def test_no_workspace_says_workspace_not_expired(self, _mock: object) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        creds = _make_credentials()

        with pytest.raises(PlatformAPIError, match="No workspace selected"):
            platform_request("/api/test", credentials=creds)


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
                "No workspace selected. Run 'osmosis workspace' to select a workspace."
            ),
        ):
            code = main([])

        assert code == 1
        assert "No workspace selected" in capsys.readouterr().err

    @patch("osmosis_ai.cli.main._registered", True)
    def test_auth_expired_shows_session_expired(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from osmosis_ai.cli.main import main

        with patch(
            "osmosis_ai.cli.main.app",
            side_effect=AuthenticationExpiredError("expired"),
        ):
            code = main([])

        assert code == 1
        captured = capsys.readouterr().err
        assert (
            "session has expired" in captured.lower() or "expired" in captured.lower()
        )
