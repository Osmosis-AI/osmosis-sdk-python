"""Tests for osmosis_ai.platform.auth.flow - login orchestration."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from osmosis_ai.platform.auth.config import PLATFORM_URL
from osmosis_ai.platform.auth.credentials import (
    OrganizationInfo,
    UserInfo,
    WorkspaceCredentials,
)
from osmosis_ai.platform.auth.flow import (
    LoginError,
    LoginResult,
    VerifyResult,
    _build_login_url,
    _generate_state,
    _get_device_name,
    _verify_and_get_user_info,
    login,
)

# ---------------------------------------------------------------------------
# _generate_state
# ---------------------------------------------------------------------------


class TestGenerateState:
    def test_returns_string(self) -> None:
        result = _generate_state()
        assert isinstance(result, str)

    def test_returns_non_empty(self) -> None:
        result = _generate_state()
        assert len(result) > 0

    def test_returns_unique_values(self) -> None:
        states = {_generate_state() for _ in range(50)}
        assert len(states) == 50, "Each generated state should be unique"


# ---------------------------------------------------------------------------
# _get_device_name
# ---------------------------------------------------------------------------


class TestGetDeviceName:
    def test_returns_hostname(self) -> None:
        name = _get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_matches_socket_gethostname(self) -> None:
        expected = socket.gethostname()
        assert _get_device_name() == expected

    def test_falls_back_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            socket, "gethostname", lambda: (_ for _ in ()).throw(OSError("mocked"))
        )
        assert _get_device_name() == "Unknown"


# ---------------------------------------------------------------------------
# _build_login_url
# ---------------------------------------------------------------------------


class TestBuildLoginUrl:
    def test_url_starts_with_platform_url(self) -> None:
        url = _build_login_url(state="abc123", port=8976)
        assert url.startswith(PLATFORM_URL)

    def test_url_path_is_cli_auth(self) -> None:
        url = _build_login_url(state="abc123", port=8976)
        parsed = urlparse(url)
        assert parsed.path == "/cli-auth"

    def test_query_params_include_state(self) -> None:
        url = _build_login_url(state="mystate", port=8976)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert params["state"] == ["mystate"]

    def test_query_params_include_port(self) -> None:
        url = _build_login_url(state="s", port=9999)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert params["port"] == ["9999"]

    def test_query_params_include_redirect_uri(self) -> None:
        url = _build_login_url(state="s", port=8980)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert params["redirect_uri"] == ["http://localhost:8980/callback"]

    def test_query_params_include_device_name(self) -> None:
        url = _build_login_url(state="s", port=8976)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert "device_name" in params
        assert len(params["device_name"][0]) > 0


# ---------------------------------------------------------------------------
# _verify_and_get_user_info
# ---------------------------------------------------------------------------


def _make_verify_response(
    *,
    valid: bool = True,
    user: dict[str, Any] | None = None,
    organization: dict[str, Any] | None = None,
    expires_at: str | None = None,
    token_id: str | None = "tok_abc",
) -> bytes:
    """Build a JSON response body for the /api/cli/verify endpoint."""
    data: dict[str, Any] = {"valid": valid}
    if user is not None:
        data["user"] = user
    else:
        data["user"] = {"id": "u1", "email": "u@test.com", "name": "Test User"}
    if organization is not None:
        data["organization"] = organization
    else:
        data["organization"] = {"id": "o1", "name": "TestOrg", "role": "owner"}
    if expires_at is not None:
        data["expires_at"] = expires_at
    if token_id is not None:
        data["token_id"] = token_id
    return json.dumps(data).encode()


class TestVerifyAndGetUserInfo:
    def test_successful_verification(self) -> None:
        expires_str = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        body = _make_verify_response(expires_at=expires_str)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _verify_and_get_user_info("test-token")

        assert isinstance(result, VerifyResult)
        assert isinstance(result.user, UserInfo)
        assert result.user.email == "u@test.com"
        assert isinstance(result.organization, OrganizationInfo)
        assert result.organization.name == "TestOrg"
        assert result.expires_at.tzinfo is not None
        assert result.token_id == "tok_abc"
        assert result.projects is None

    def test_verification_with_no_expires_at_defaults_to_90_days(self) -> None:
        body = _make_verify_response(expires_at=None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        before = datetime.now(timezone.utc)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _verify_and_get_user_info("token")
        after = datetime.now(timezone.utc)

        assert result.expires_at >= before + timedelta(days=89)
        assert result.expires_at <= after + timedelta(days=91)

    def test_invalid_token_raises_login_error(self) -> None:
        body = _make_verify_response(valid=False)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="Token verification failed"):
                _verify_and_get_user_info("bad-token")

    def test_http_401_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Invalid or expired token"):
                _verify_and_get_user_info("expired-token")

    def test_http_500_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Verification failed: HTTP 500"):
                _verify_and_get_user_info("token")

    def test_network_error_raises_login_error(self) -> None:
        error = URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Could not connect to platform"):
                _verify_and_get_user_info("token")

    def test_invalid_json_raises_login_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="Invalid response from platform"):
                _verify_and_get_user_info("token")

    def test_naive_expires_at_raises_login_error(self) -> None:
        """A naive (non-timezone-aware) expires_at should trigger a LoginError."""
        naive_iso = datetime.now().isoformat()  # no tz info
        body = _make_verify_response(expires_at=naive_iso)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="expected timezone-aware"):
                _verify_and_get_user_info("token")

    def test_missing_user_fields_fallback_to_defaults(self) -> None:
        """Missing user/org fields should fall back to empty strings."""
        expires_str = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        body = _make_verify_response(
            user={},
            organization={},
            expires_at=expires_str,
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _verify_and_get_user_info("token")

        assert result.user.id == ""
        assert result.user.email == ""
        assert result.user.name is None
        assert result.organization.id == ""
        assert result.organization.name == ""
        assert result.organization.role == "member"

    def test_no_token_id_in_response(self) -> None:
        expires_str = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        body = _make_verify_response(
            expires_at=expires_str,
            token_id=None,
        )
        # Remove token_id from payload
        data = json.loads(body)
        data.pop("token_id", None)
        body = json.dumps(data).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _verify_and_get_user_info("token")

        assert result.token_id is None


# ---------------------------------------------------------------------------
# login() full flow
# ---------------------------------------------------------------------------


class TestLogin:
    """Tests for the top-level login() orchestrator."""

    def _stub_verify_response(self) -> MagicMock:
        """Build a mock urlopen response for successful verification."""
        expires_str = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        body = _make_verify_response(expires_at=expires_str)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_no_available_port_raises_login_error(self) -> None:
        with patch(
            "osmosis_ai.platform.auth.flow.find_available_port", return_value=None
        ):
            with pytest.raises(LoginError, match="No available port"):
                login()

    def test_successful_login_flow(self) -> None:
        """Full happy path: port found -> browser opens -> callback -> verify -> save."""
        mock_verify_resp = self._stub_verify_response()

        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("test-token-123", None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser") as mock_wb,
            patch("urllib.request.urlopen", return_value=mock_verify_resp),
            patch("osmosis_ai.platform.auth.flow.save_credentials") as mock_save,
        ):
            mock_wb.open.return_value = True
            result = login()

        assert isinstance(result, LoginResult)
        assert result.user.email == "u@test.com"
        assert result.organization.name == "TestOrg"
        mock_save.assert_called_once()

    def test_no_browser_mode_does_not_open_browser(self) -> None:
        mock_verify_resp = self._stub_verify_response()

        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("test-token", None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser") as mock_wb,
            patch("urllib.request.urlopen", return_value=mock_verify_resp),
            patch("osmosis_ai.platform.auth.flow.save_credentials"),
        ):
            login(no_browser=True)

        mock_wb.open.assert_not_called()

    def test_callback_returns_error(self) -> None:
        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = (None, "User denied")
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = False

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser"),
        ):
            with pytest.raises(LoginError, match="Authentication failed: User denied"):
                login()

    def test_callback_returns_no_token(self) -> None:
        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = (None, None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = False

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser"),
        ):
            with pytest.raises(LoginError, match="No token received"):
                login()

    def test_verification_failure_propagates(self) -> None:
        """If _verify_and_get_user_info fails, login() should propagate LoginError."""
        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("bad-token", None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        verify_error = HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser"),
            patch("urllib.request.urlopen", side_effect=verify_error),
        ):
            with pytest.raises(LoginError, match="Invalid or expired token"):
                login()

        # set_verification_result should have been called with success=False
        mock_server.set_verification_result.assert_called_once()
        call_kwargs = mock_server.set_verification_result.call_args
        assert call_kwargs[1].get("success") is False or call_kwargs[0][0] is False

    def test_server_shutdown_called_in_finally(self) -> None:
        """The server should always be cleaned up even on failure."""
        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = (None, "error")
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = False
        mock_server._shutdown_event = MagicMock()

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser"),
        ):
            with pytest.raises(LoginError):
                login()

        mock_server.server_close.assert_called_once()

    def test_browser_open_failure_does_not_raise(self) -> None:
        """If webbrowser.open returns False, login should still proceed."""
        mock_verify_resp = self._stub_verify_response()

        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("test-token", None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser") as mock_wb,
            patch("urllib.request.urlopen", return_value=mock_verify_resp),
            patch("osmosis_ai.platform.auth.flow.save_credentials"),
        ):
            mock_wb.open.return_value = False
            result = login()

        assert isinstance(result, LoginResult)

    def test_revoked_previous_tokens_count(self) -> None:
        """revoked_count from server should propagate to LoginResult."""
        mock_verify_resp = self._stub_verify_response()

        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("test-token", None)
        mock_server.revoked_count = 3
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser") as mock_wb,
            patch("urllib.request.urlopen", return_value=mock_verify_resp),
            patch("osmosis_ai.platform.auth.flow.save_credentials"),
        ):
            mock_wb.open.return_value = True
            result = login()

        assert result.revoked_previous_tokens == 3

    def test_credentials_saved_with_correct_token(self) -> None:
        """Credentials passed to save_credentials should contain the actual token."""
        mock_verify_resp = self._stub_verify_response()

        mock_server = MagicMock()
        mock_server.wait_for_callback.return_value = ("the-real-token", None)
        mock_server.revoked_count = 0
        mock_server._verification_event = MagicMock()
        mock_server._verification_event.is_set.return_value = True

        with (
            patch(
                "osmosis_ai.platform.auth.flow.find_available_port", return_value=8976
            ),
            patch(
                "osmosis_ai.platform.auth.flow.LocalAuthServer",
                return_value=mock_server,
            ),
            patch("osmosis_ai.platform.auth.flow.webbrowser") as mock_wb,
            patch("urllib.request.urlopen", return_value=mock_verify_resp),
            patch("osmosis_ai.platform.auth.flow.save_credentials") as mock_save,
        ):
            mock_wb.open.return_value = True
            login()

        saved_creds: WorkspaceCredentials = mock_save.call_args[0][0]
        assert saved_creds.access_token == "the-real-token"
        assert saved_creds.token_type == "Bearer"
