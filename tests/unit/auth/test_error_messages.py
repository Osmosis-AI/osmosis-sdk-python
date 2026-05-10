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


def test_reset_session_does_not_delete_project_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osmosis_ai.platform.auth.local_config import reset_session

    legacy_config = tmp_path / ".config" / "osmosis" / "config.json"
    legacy_cache = tmp_path / ".config" / "osmosis" / "cache"
    legacy_config.parent.mkdir(parents=True)
    legacy_cache.mkdir()
    legacy_config.write_text(
        '{"active_workspace":{"id":"legacy","name":"legacy"}}',
        encoding="utf-8",
    )

    project_mapping = tmp_path / ".osmosis" / "config.json"
    project_mapping.parent.mkdir()
    project_mapping.write_text('{"version":1,"platforms":{}}', encoding="utf-8")

    monkeypatch.setattr(
        "osmosis_ai.platform.auth.local_config.CONFIG_FILE",
        legacy_config,
    )
    monkeypatch.setattr("osmosis_ai.platform.auth.local_config.CACHE_DIR", legacy_cache)
    monkeypatch.setattr(
        "osmosis_ai.platform.auth.credentials.delete_credentials", lambda: True
    )
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.project_mapping.CONFIG_FILE",
        project_mapping,
    )

    reset_session()

    assert not legacy_config.exists()
    assert project_mapping.exists()


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


class TestRequireAuthMessages:
    """Test _require_auth() no-arg calls require linked-project migration."""

    @patch("osmosis_ai.platform.cli.utils.load_credentials", return_value=None)
    def test_no_arg_requires_linked_project_before_login(
        self, cred_mock: object
    ) -> None:
        """No-arg _require_auth is no longer an auth/workspace selector."""
        from osmosis_ai.platform.cli.utils import _require_auth

        with pytest.raises(CLIError, match="linked Osmosis project"):
            _require_auth()

        cred_mock.assert_not_called()

    @patch("osmosis_ai.platform.cli.utils.platform_call")
    def test_no_arg_does_not_auto_select_workspace(
        self, platform_call_mock: object
    ) -> None:
        """No-arg _require_auth must not fall back to global active workspace."""
        from osmosis_ai.platform.cli.utils import _require_auth

        with pytest.raises(CLIError, match="linked Osmosis project"):
            _require_auth()

        platform_call_mock.assert_not_called()

    @patch("osmosis_ai.platform.cli.utils.load_credentials", return_value=None)
    def test_explicit_workspace_name_still_checks_login(self, _mock: object) -> None:
        """Bootstrap flows passing workspace_name keep credential validation."""
        from osmosis_ai.platform.cli.utils import _require_auth

        with pytest.raises(CLIError, match="Not logged in"):
            _require_auth(workspace_name="bootstrap-workspace")

    @patch("osmosis_ai.platform.cli.utils.load_credentials")
    def test_explicit_workspace_name_returns_credentials(
        self, mock_load: object
    ) -> None:
        """Bootstrap flows can still pass an already-resolved workspace name."""
        from osmosis_ai.platform.cli.utils import _require_auth

        creds = _make_credentials()
        mock_load.return_value = creds

        workspace_name, resolved_credentials = _require_auth(
            workspace_name="bootstrap-workspace"
        )

        assert workspace_name == "bootstrap-workspace"
        assert resolved_credentials is creds


class TestPlatformRequestMessages:
    """Test platform_request() error messages for each failure mode."""

    @patch(
        "osmosis_ai.platform.auth.platform_client.load_credentials", return_value=None
    )
    def test_no_credentials_does_not_say_expired(self, _mock: object) -> None:
        from osmosis_ai.platform.auth.platform_client import platform_request

        with pytest.raises(CLIError, match="Not logged in"):
            platform_request("/api/test")

    def test_workspace_required_ignores_legacy_active_workspace_fallback(
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

        with pytest.raises(PlatformAPIError, match="explicit workspace_id"):
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
                "This project is not linked to an Osmosis workspace for the "
                "current platform. Run 'osmosis project link' from the project root."
            ),
        ):
            code = main([])

        assert code == 1
        assert "This project is not linked" in capsys.readouterr().err

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
