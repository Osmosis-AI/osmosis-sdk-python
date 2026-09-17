"""Tests for osmosis_ai.platform.auth.flow."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
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


def _make_device_code_response(include_complete_uri: bool = True) -> bytes:
    data = {
        "device_code": "device_abc123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://platform.example.test/device",
        "expires_in": 600,
        "interval": 5,
    }
    if include_complete_uri:
        data["verification_uri_complete"] = (
            "https://platform.example.test/device?code=ABCD-1234"
        )
    return json.dumps(data).encode()


def _make_token_response(**overrides: object) -> bytes:
    """Build a flat token response dict (matches device/token endpoint)."""
    expires_str = (datetime.now(UTC) + timedelta(days=90)).isoformat()
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
        assert (
            result.verification_uri_complete
            == "https://platform.example.test/device?code=ABCD-1234"
        )

    def test_missing_complete_uri_defaults_to_none(self) -> None:
        body = _make_device_code_response(include_complete_uri=False)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            result = request_device_code()

        assert result.verification_uri_complete is None

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
            patch("webbrowser.open", return_value=True) as mock_open,
        ):
            mock_stdin.isatty.return_value = True
            result, creds = device_login(timeout=10.0)

        mock_open.assert_called_once_with(
            "https://platform.example.test/device?code=ABCD-1234"
        )
        assert result.user.email == "u@test.com"
        assert creds.access_token == "jwt-user-token"
        assert creds.token_id == "tok_123"
        assert creds.expires_at.tzinfo is not None

    def test_browser_falls_back_to_plain_uri_without_complete(self) -> None:
        device_resp_body = _make_device_code_response(include_complete_uri=False)
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
            patch("webbrowser.open", return_value=True) as mock_open,
        ):
            mock_stdin.isatty.return_value = True
            device_login(timeout=10.0)

        mock_open.assert_called_once_with("https://platform.example.test/device")


@pytest.mark.parametrize("polling", [False, True])
@pytest.mark.parametrize(
    "error_value", ["workspace_unavailable", {"code": "workspace_unavailable"}]
)
@pytest.mark.parametrize("status", [400, 409])
def test_device_login_errors_surface_server_explanation(
    polling: bool,
    error_value: object,
    status: int,
) -> None:
    body = {
        "error": error_value,
        "message": "Ask your workspace administrator to restore access.",
    }
    error = HTTPError(
        url="http://test",
        code=status,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(json.dumps(body).encode()),
    )
    with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error) as request:
        with pytest.raises(
            LoginError, match="Ask your workspace administrator to restore access"
        ) as caught:
            if polling:
                poll_device_token("device_abc", interval=1, timeout=10)
            else:
                request_device_code()
    request.assert_called_once()
    assert caught.value.status_code == status


@pytest.mark.parametrize("polling", [False, True])
def test_device_login_deployment_conflict_gets_cli_guidance(polling: bool) -> None:
    body = {
        "error": "deployment_conflict",
        "message": "We\u2019ve updated the app. Reload to continue.",
    }
    error = HTTPError(
        url="http://test",
        code=409,
        msg="Conflict",
        hdrs=None,
        fp=BytesIO(json.dumps(body).encode()),
    )
    with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
        with pytest.raises(LoginError, match="try again shortly") as caught:
            if polling:
                poll_device_token("device_abc", interval=1, timeout=10)
            else:
                request_device_code()
    assert "Reload" not in str(caught.value)
    assert caught.value.status_code == 409
    # The legacy ``error`` field is the only place the platform code lives here.
    assert caught.value.code == "deployment_conflict"
    assert caught.value.details == body


@pytest.mark.parametrize(
    ("error_code", "message"),
    [
        ("expired_token", "Device code expired"),
        ("access_denied", "Authorization was denied"),
        ("invalid_grant", "Polling failed"),
    ],
)
def test_poll_terminal_errors_carry_status_and_code(
    error_code: str, message: str
) -> None:
    error = HTTPError(
        url="http://test",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(json.dumps({"error": error_code}).encode()),
    )
    with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
        with pytest.raises(LoginError, match=message) as caught:
            poll_device_token("device_abc", interval=1, timeout=10)
    assert caught.value.status_code == 400
    assert caught.value.code == error_code
