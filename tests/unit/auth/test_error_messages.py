"""Tests that verify each auth error state produces the correct user-facing message.

These are integration-style tests: they call the CLI entry point and check
stderr output, ensuring the exception type -> main.py handler -> message chain
is correct end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.credentials import Credentials, UserInfo
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
)
from osmosis_ai.platform.constants import MSG_ENV_TOKEN_INVALID


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


def test_reset_session_deletes_legacy_auth_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.platform.auth.local_config import reset_session

    legacy_config = tmp_path / ".config" / "osmosis" / "config.json"
    legacy_config.parent.mkdir(parents=True)
    legacy_config.write_text(
        '{"active_workspace":{"id":"legacy","name":"legacy"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.CONFIG_FILE",
        legacy_config,
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials", lambda: True
    )

    reset_session()

    assert not legacy_config.exists()


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

        with pytest.raises(AuthenticationExpiredError) as exc_info:
            require_credentials()

        assert "session has expired" in str(exc_info.value)
        assert "osmosis auth login" in str(exc_info.value)

    @patch("osmosis_ai.platform.cli.utils.load_credentials")
    def test_valid_credentials_returns_them(self, mock_load: object) -> None:
        from osmosis_ai.platform.cli.utils import require_credentials

        creds = _make_credentials()
        mock_load.return_value = creds

        result = require_credentials()
        assert result is creds


def test_global_active_workspace_helpers_are_not_public_context_api() -> None:
    """Production modules must not expose global active workspace context APIs."""
    import osmosis_ai.platform.auth as auth_module
    import osmosis_ai.platform.auth.local_config as local_config
    import osmosis_ai.platform.auth.platform_client as platform_client
    import osmosis_ai.platform.cli.utils as cli_utils

    removed_symbols = [
        (auth_module, "ensure_active_workspace"),
        (auth_module, "get_active_workspace"),
        (auth_module, "get_active_workspace_id"),
        (local_config, "get_active_workspace"),
        (local_config, "get_active_workspace_id"),
        (local_config, "get_active_workspace_name"),
        (local_config, "set_active_workspace"),
        (local_config, "clear_active_workspace"),
        (platform_client, "ensure_active_workspace"),
        (platform_client, "get_active_workspace_id"),
        (cli_utils, "_get_active_workspace_name"),
    ]
    for module, symbol in removed_symbols:
        assert not hasattr(module, symbol), f"{module.__name__}.{symbol} remains"


class TestPlatformRequestMessages:
    """Test platform_request() error messages for each failure mode."""

    @patch(
        "osmosis_ai.platform.auth.platform_client.load_credentials", return_value=None
    )
    def test_no_credentials_does_not_say_expired(self, _mock: object) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        with pytest.raises(CLIError, match="Not logged in"):
            platform_request("/api/test")

    def test_git_scope_required_ignores_legacy_active_workspace_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        legacy_config = tmp_path / "config.json"
        legacy_config.write_text(
            '{"active_workspace":{"id":"legacy-ws","name":"legacy-workspace"}}',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "osmosis_ai.platform.auth.local_config.CONFIG_FILE",
            legacy_config,
        )
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError, match="explicit git_identity"):
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
