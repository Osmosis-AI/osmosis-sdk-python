"""Tests for osmosis_ai.auth.platform_client."""

from __future__ import annotations

import http.client
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from osmosis_ai.auth.credentials import OrganizationInfo, UserInfo, WorkspaceCredentials
from osmosis_ai.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    _handle_401_and_cleanup,
    platform_request,
)

# =============================================================================
# Helper: Create real WorkspaceCredentials for testing
# =============================================================================


def _make_credentials(
    access_token: str = "test-token-abc123",
    org_name: str = "TestOrg",
) -> WorkspaceCredentials:
    """Create valid WorkspaceCredentials for testing."""
    now = datetime.now(timezone.utc)
    return WorkspaceCredentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=now + timedelta(days=30),
        user=UserInfo(id="user_1", email="user@example.com", name="Test User"),
        organization=OrganizationInfo(id="org_1", name=org_name, role="member"),
        created_at=now,
    )


def _make_http_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock HTTP response that behaves like urlopen() return value."""
    body = json.dumps(data).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(
    code: int, body: str = "", url: str = "https://platform.osmosis.ai/api/test"
) -> HTTPError:
    """Create an HTTPError with readable body."""
    fp = BytesIO(body.encode("utf-8")) if body else BytesIO(b"")
    return HTTPError(
        url=url, code=code, msg=http.client.responses.get(code, "Error"), hdrs={}, fp=fp
    )


# =============================================================================
# Exception Classes Tests
# =============================================================================


class TestExceptionClasses:
    """Tests for AuthenticationExpiredError and PlatformAPIError."""

    def test_authentication_expired_error_is_exception(self) -> None:
        """Verify AuthenticationExpiredError is a proper Exception subclass."""
        err = AuthenticationExpiredError("session expired")
        assert isinstance(err, Exception)
        assert str(err) == "session expired"

    def test_platform_api_error_without_status_code(self) -> None:
        """Verify PlatformAPIError can be created without a status code."""
        err = PlatformAPIError("connection failed")
        assert str(err) == "connection failed"
        assert err.status_code is None

    def test_platform_api_error_with_status_code(self) -> None:
        """Verify PlatformAPIError stores the HTTP status code."""
        err = PlatformAPIError("not found", status_code=404)
        assert err.status_code == 404
        assert "not found" in str(err)


# =============================================================================
# _handle_401_and_cleanup Tests
# =============================================================================


class TestHandle401AndCleanup:
    """Tests for the _handle_401_and_cleanup function."""

    @patch("osmosis_ai.auth.platform_client.delete_workspace_credentials")
    @patch("osmosis_ai.auth.platform_client.get_active_workspace")
    def test_deletes_active_workspace_credentials(
        self, mock_get_ws: MagicMock, mock_delete: MagicMock
    ) -> None:
        """Verify 401 handler deletes credentials for the active workspace."""
        mock_get_ws.return_value = "MyWorkspace"

        with pytest.raises(AuthenticationExpiredError, match="expired or been revoked"):
            _handle_401_and_cleanup()

        mock_delete.assert_called_once_with("MyWorkspace")

    @patch("osmosis_ai.auth.platform_client.delete_workspace_credentials")
    @patch("osmosis_ai.auth.platform_client.get_active_workspace")
    def test_no_active_workspace_skips_delete(
        self, mock_get_ws: MagicMock, mock_delete: MagicMock
    ) -> None:
        """Verify 401 handler does not call delete when no workspace is active."""
        mock_get_ws.return_value = None

        with pytest.raises(AuthenticationExpiredError, match="osmosis login"):
            _handle_401_and_cleanup()

        mock_delete.assert_not_called()

    @patch("osmosis_ai.auth.platform_client.delete_workspace_credentials")
    @patch("osmosis_ai.auth.platform_client.get_active_workspace")
    def test_always_raises_even_after_cleanup(
        self, mock_get_ws: MagicMock, mock_delete: MagicMock
    ) -> None:
        """Verify 401 handler always raises AuthenticationExpiredError."""
        mock_get_ws.return_value = "SomeWorkspace"
        mock_delete.return_value = True

        with pytest.raises(AuthenticationExpiredError):
            _handle_401_and_cleanup()


# =============================================================================
# platform_request Tests
# =============================================================================


class TestPlatformRequest:
    """Tests for the platform_request function."""

    # -------------------------------------------------------------------------
    # Credential Loading
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.load_credentials")
    def test_raises_when_no_credentials_found(self, mock_load: MagicMock) -> None:
        """Verify AuthenticationExpiredError when no credentials exist."""
        mock_load.return_value = None

        with pytest.raises(AuthenticationExpiredError, match="No valid credentials"):
            platform_request("/api/test")

    @patch("osmosis_ai.auth.platform_client.urlopen")
    @patch("osmosis_ai.auth.platform_client.load_credentials")
    def test_uses_loaded_credentials_when_none_provided(
        self, mock_load: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        """Verify credentials are loaded from storage when not explicitly provided."""
        creds = _make_credentials(access_token="loaded-token")
        mock_load.return_value = creds
        mock_urlopen.return_value = _make_http_response({"ok": True})

        platform_request("/api/test")

        mock_load.assert_called_once()
        # Verify the loaded token was used in the request
        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("Authorization") == "Bearer loaded-token"

    @patch("osmosis_ai.auth.platform_client.urlopen")
    @patch("osmosis_ai.auth.platform_client.load_credentials")
    def test_uses_explicit_credentials_when_provided(
        self, mock_load: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        """Verify explicit credentials bypass loading from storage."""
        explicit_creds = _make_credentials(access_token="explicit-token")
        mock_urlopen.return_value = _make_http_response({"ok": True})

        platform_request("/api/test", credentials=explicit_creds)

        mock_load.assert_not_called()
        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("Authorization") == "Bearer explicit-token"

    # -------------------------------------------------------------------------
    # Request Construction
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_constructs_correct_url(self, mock_urlopen: MagicMock) -> None:
        """Verify the full URL is built from PLATFORM_URL + endpoint."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        with patch(
            "osmosis_ai.auth.platform_client.PLATFORM_URL", "https://test.osmosis.ai"
        ):
            platform_request("/api/v1/verify", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.full_url == "https://test.osmosis.ai/api/v1/verify"

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_sets_required_headers(self, mock_urlopen: MagicMock) -> None:
        """Verify Authorization, Content-Type, and User-Agent headers are set."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials(access_token="my-token")

        platform_request("/api/test", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("Authorization") == "Bearer my-token"
        assert request_obj.get_header("Content-type") == "application/json"
        assert "osmosis-cli/" in request_obj.get_header("User-agent")

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_custom_headers_are_merged(self, mock_urlopen: MagicMock) -> None:
        """Verify additional headers are merged into the request."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request(
            "/api/test",
            headers={"X-Custom": "value123"},
            credentials=creds,
        )

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("X-custom") == "value123"

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_get_request_has_no_body(self, mock_urlopen: MagicMock) -> None:
        """Verify GET requests send no request body."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", method="GET", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.data is None
        assert request_obj.get_method() == "GET"

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_post_request_sends_json_body(self, mock_urlopen: MagicMock) -> None:
        """Verify POST requests with data encode body as JSON."""
        mock_urlopen.return_value = _make_http_response({"created": True})
        creds = _make_credentials()
        payload = {"host": "10.0.0.1", "port": 8080}

        platform_request(
            "/api/register", method="POST", data=payload, credentials=creds
        )

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_method() == "POST"
        body = json.loads(request_obj.data.decode())
        assert body == {"host": "10.0.0.1", "port": 8080}

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_timeout_is_passed_to_urlopen(self, mock_urlopen: MagicMock) -> None:
        """Verify the timeout parameter is forwarded to urlopen."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", timeout=5.0, credentials=creds)

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 5.0

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_default_timeout_is_30(self, mock_urlopen: MagicMock) -> None:
        """Verify default timeout is 30 seconds."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", credentials=creds)

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 30.0

    # -------------------------------------------------------------------------
    # Successful Response Handling
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_returns_parsed_json_response(self, mock_urlopen: MagicMock) -> None:
        """Verify the response JSON is parsed and returned."""
        expected = {"id": "srv_1", "status": "healthy"}
        mock_urlopen.return_value = _make_http_response(expected)
        creds = _make_credentials()

        result = platform_request("/api/test", credentials=creds)

        assert result == expected

    # -------------------------------------------------------------------------
    # HTTP Error Handling
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client._handle_401_and_cleanup")
    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_401_triggers_cleanup_and_raises(
        self, mock_urlopen: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        """Verify 401 response triggers credential cleanup."""
        mock_urlopen.side_effect = _make_http_error(401, "Unauthorized")
        mock_cleanup.side_effect = AuthenticationExpiredError("expired")
        creds = _make_credentials()

        with pytest.raises(AuthenticationExpiredError):
            platform_request("/api/test", credentials=creds)

        mock_cleanup.assert_called_once()

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_non_401_http_error_raises_platform_api_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify non-401 HTTP errors raise PlatformAPIError with status code."""
        mock_urlopen.side_effect = _make_http_error(500, "Internal Server Error")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        assert exc_info.value.status_code == 500
        assert "HTTP 500" in str(exc_info.value)

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_http_error_includes_response_body_in_message(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify the error response body is captured in the PlatformAPIError message."""
        error_body = '{"detail": "Agent not found"}'
        mock_urlopen.side_effect = _make_http_error(404, error_body)
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        assert "Agent not found" in str(exc_info.value)
        assert exc_info.value.status_code == 404

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_http_error_truncates_long_response_body(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify response bodies longer than 500 chars are truncated."""
        long_body = "x" * 600
        mock_urlopen.side_effect = _make_http_error(500, long_body)
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        msg = str(exc_info.value)
        assert "(truncated)" in msg

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_http_error_with_empty_body(self, mock_urlopen: MagicMock) -> None:
        """Verify HTTP errors with empty body still produce a useful message."""
        mock_urlopen.side_effect = _make_http_error(502, "")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        assert "HTTP 502" in str(exc_info.value)
        assert exc_info.value.status_code == 502

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_http_error_with_unreadable_body_does_not_crash(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify HTTP error handling is robust when body read fails."""
        err = _make_http_error(503, "error")
        # Make the read() call raise, simulating a broken stream
        err.read = MagicMock(side_effect=OSError("stream broken"))
        mock_urlopen.side_effect = err
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        # Should still get a PlatformAPIError with the status code
        assert exc_info.value.status_code == 503
        assert "HTTP 503" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # URLError (Connection Errors)
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_url_error_raises_platform_api_error(self, mock_urlopen: MagicMock) -> None:
        """Verify URLError is wrapped in PlatformAPIError."""
        mock_urlopen.side_effect = URLError("Name resolution failed")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError, match="Connection error"):
            platform_request("/api/test", credentials=creds)

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_url_error_does_not_include_status_code(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify URLError-based PlatformAPIError has no status code."""
        mock_urlopen.side_effect = URLError("Connection refused")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        assert exc_info.value.status_code is None

    # -------------------------------------------------------------------------
    # JSON Decode Errors
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.urlopen")
    def test_invalid_json_response_raises_platform_api_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify non-JSON responses raise PlatformAPIError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>Not JSON</html>"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError, match="Invalid JSON response"):
            platform_request("/api/test", credentials=creds)

    # -------------------------------------------------------------------------
    # Integration-style: Credential flow
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.auth.platform_client.urlopen")
    @patch("osmosis_ai.auth.platform_client.load_credentials")
    def test_explicit_none_credentials_triggers_load(
        self, mock_load: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        """Verify passing credentials=None explicitly still loads from storage."""
        creds = _make_credentials()
        mock_load.return_value = creds
        mock_urlopen.return_value = _make_http_response({"ok": True})

        platform_request("/api/test", credentials=None)

        mock_load.assert_called_once()

    @patch("osmosis_ai.auth.platform_client.load_credentials")
    def test_load_returns_none_raises_auth_error(self, mock_load: MagicMock) -> None:
        """Verify AuthenticationExpiredError when load_credentials returns None."""
        mock_load.return_value = None

        with pytest.raises(AuthenticationExpiredError, match="No valid credentials"):
            platform_request("/api/test", credentials=None)
