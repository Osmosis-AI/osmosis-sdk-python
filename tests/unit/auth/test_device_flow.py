"""Tests for osmosis_ai.platform.auth.flow."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from osmosis_ai.platform.auth.flow import (
    DeviceCodeResponse,
    LoginError,
    device_login,
    poll_device_token,
    request_device_code,
)


def _make_device_code_response() -> bytes:
    data = {
        "device_code": "device_abc123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://platform.osmosis.ai/device",
        "verification_uri_complete": "https://platform.osmosis.ai/device?code=ABCD-1234",
        "expires_in": 600,
        "interval": 5,
    }
    return json.dumps(data).encode()


def _make_token_response(**overrides: object) -> bytes:
    """Build a flat token response dict (matches device/token endpoint)."""
    expires_str = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
    data: dict[str, object] = {
        "token": "jwt-user-token",
        "expires_at": expires_str,
        "token_id": "tok_123",
        "user": {"id": "u1", "email": "u@test.com", "name": "Test"},
    }
    data.update(overrides)
    return json.dumps(data).encode()


class TestRequestDeviceCode:
    def test_successful_request(self) -> None:
        body = _make_device_code_response()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            result = request_device_code()

        assert isinstance(result, DeviceCodeResponse)
        assert result.device_code == "device_abc123"
        assert result.user_code == "ABCD-1234"

    def test_http_error_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test", code=500, msg="Server Error", hdrs=None, fp=None
        )
        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="internal error"):
                request_device_code()

    def test_network_error_raises_login_error(self) -> None:
        error = URLError(reason="Connection refused")
        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Could not connect to platform"):
                request_device_code()

    def test_invalid_json_raises_login_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="Invalid response from platform"):
                request_device_code()


class TestPollDeviceToken:
    def test_pending_then_success(self) -> None:
        pending_error = HTTPError(
            url="http://test", code=400, msg="Bad Request", hdrs=None, fp=None
        )
        pending_error.read = MagicMock(
            return_value=json.dumps({"error": "authorization_pending"}).encode()
        )

        success_resp = MagicMock()
        success_resp.read.return_value = _make_token_response()
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "osmosis_ai.platform.auth.flow.urlopen",
            side_effect=[pending_error, success_resp],
        ):
            with patch("osmosis_ai.platform.auth.flow.time.sleep"):
                result = poll_device_token("device_abc", interval=1, timeout=10.0)

        # Returns full dict now, not just workspaces
        assert "token" in result
        assert result["token"] == "jwt-user-token"

    def test_expired_token_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test", code=400, msg="Bad Request", hdrs=None, fp=None
        )
        error.read = MagicMock(
            return_value=json.dumps({"error": "expired_token"}).encode()
        )

        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Device code expired"):
                poll_device_token("device_abc", interval=1, timeout=10.0)

    def test_access_denied_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test", code=400, msg="Bad Request", hdrs=None, fp=None
        )
        error.read = MagicMock(
            return_value=json.dumps({"error": "access_denied"}).encode()
        )

        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Authorization was denied"):
                poll_device_token("device_abc", interval=1, timeout=10.0)

    def test_slow_down_increases_interval(self) -> None:
        slow_down_error = HTTPError(
            url="http://test", code=400, msg="Bad Request", hdrs=None, fp=None
        )
        slow_down_error.read = MagicMock(
            return_value=json.dumps({"error": "slow_down"}).encode()
        )

        success_resp = MagicMock()
        success_resp.read.return_value = _make_token_response()
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch(
            "osmosis_ai.platform.auth.flow.urlopen",
            side_effect=[slow_down_error, success_resp],
        ):
            with patch(
                "osmosis_ai.platform.auth.flow.time.sleep",
                side_effect=mock_sleep,
            ):
                poll_device_token("device_abc", interval=5, timeout=30.0)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == 10

    def test_timeout_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test", code=400, msg="Bad Request", hdrs=None, fp=None
        )
        error.read = MagicMock(
            return_value=json.dumps({"error": "authorization_pending"}).encode()
        )

        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with patch("osmosis_ai.platform.auth.flow.time.sleep"):
                with patch(
                    "osmosis_ai.platform.auth.flow.time.monotonic",
                    side_effect=[0, 100, 200],
                ):
                    with pytest.raises(LoginError, match="timed out"):
                        poll_device_token("device_abc", interval=1, timeout=1.0)


class TestDeviceLogin:
    def test_full_device_login_flow(self) -> None:
        device_resp_body = _make_device_code_response()
        device_resp = MagicMock()
        device_resp.read.return_value = device_resp_body
        device_resp.__enter__ = MagicMock(return_value=device_resp)
        device_resp.__exit__ = MagicMock(return_value=False)

        token_resp = MagicMock()
        token_resp.read.return_value = _make_token_response()
        token_resp.__enter__ = MagicMock(return_value=token_resp)
        token_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "osmosis_ai.platform.auth.flow.urlopen",
                side_effect=[device_resp, token_resp],
            ),
            patch("osmosis_ai.platform.auth.flow.time.sleep"),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value=""),
            patch("webbrowser.open", return_value=True),
        ):
            mock_stdin.isatty.return_value = True
            result, creds = device_login(timeout=10.0)

        assert result.user.email == "u@test.com"
        assert creds.access_token == "jwt-user-token"
        assert creds.token_id == "tok_123"
        assert creds.expires_at.tzinfo is not None
