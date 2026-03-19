"""Tests for osmosis_ai.platform.auth.flow - shared login types and verification."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from osmosis_ai.platform.auth.credentials import UserInfo
from osmosis_ai.platform.auth.flow import (
    LoginError,
    VerifyResult,
    _get_device_name,
    verify_token,
)

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
# verify_token (formerly _verify_and_get_user_info)
# ---------------------------------------------------------------------------


def _make_verify_response(
    *,
    user: dict[str, Any] | None = None,
    expires_at: str | None = None,
    token_id: str | None = "tok_abc",
) -> bytes:
    """Build a JSON response body for the /api/cli/verify endpoint."""
    data: dict[str, Any] = {}
    if user is not None:
        data["user"] = user
    else:
        data["user"] = {"id": "u1", "email": "u@test.com", "name": "Test User"}
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

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            result = verify_token("test-token")

        assert isinstance(result, VerifyResult)
        assert isinstance(result.user, UserInfo)
        assert result.user.email == "u@test.com"
        assert result.expires_at.tzinfo is not None
        assert result.token_id == "tok_abc"

    def test_verification_with_no_expires_at_defaults_to_90_days(self) -> None:
        body = _make_verify_response(expires_at=None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        before = datetime.now(timezone.utc)
        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            result = verify_token("token")
        after = datetime.now(timezone.utc)

        assert result.expires_at >= before + timedelta(days=89)
        assert result.expires_at <= after + timedelta(days=91)

    def test_http_401_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Invalid or expired token"):
                verify_token("expired-token")

    def test_http_500_raises_login_error(self) -> None:
        error = HTTPError(
            url="http://test",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Verification failed: HTTP 500"):
                verify_token("token")

    def test_network_error_raises_login_error(self) -> None:
        error = URLError(reason="Connection refused")
        with patch("osmosis_ai.platform.auth.flow.urlopen", side_effect=error):
            with pytest.raises(LoginError, match="Could not connect to platform"):
                verify_token("token")

    def test_invalid_json_raises_login_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="Invalid response from platform"):
                verify_token("token")

    def test_naive_expires_at_raises_login_error(self) -> None:
        """A naive (non-timezone-aware) expires_at should trigger a LoginError."""
        naive_iso = datetime.now().isoformat()  # no tz info
        body = _make_verify_response(expires_at=naive_iso)
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match="expected timezone-aware"):
                verify_token("token")

    @pytest.mark.parametrize(
        "user_data, expected_match",
        [
            (
                {},
                "incomplete user information",
            ),
            (
                {"id": "", "email": "u@test.com", "name": "Test"},
                "incomplete user information",
            ),
            (
                {"id": "u1", "email": "", "name": "Test"},
                "incomplete user information",
            ),
        ],
        ids=[
            "missing_user_fields",
            "empty_user_id",
            "empty_user_email",
        ],
    )
    def test_incomplete_fields_raise_login_error(
        self, user_data: dict, expected_match: str
    ) -> None:
        """Incomplete user fields should raise LoginError."""
        expires_str = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        body = _make_verify_response(
            user=user_data,
            expires_at=expires_str,
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            with pytest.raises(LoginError, match=expected_match):
                verify_token("token")

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

        with patch("osmosis_ai.platform.auth.flow.urlopen", return_value=mock_resp):
            result = verify_token("token")

        assert result.token_id is None
