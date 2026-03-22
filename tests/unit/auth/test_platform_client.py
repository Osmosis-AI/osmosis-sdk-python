"""Tests for osmosis_ai.platform.auth.platform_client."""

from __future__ import annotations

import http.client
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from osmosis_ai.platform.auth.credentials import (
    Credentials,
    UserInfo,
)
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    PlatformAPIError,
    platform_request,
    revoke_cli_token,
)

# =============================================================================
# Helper: Create real Credentials for testing
# =============================================================================


def _make_credentials(
    access_token: str = "test-token-abc123",
) -> Credentials:
    """Create valid Credentials for testing."""
    now = datetime.now(timezone.utc)
    return Credentials(
        access_token=access_token,
        token_type="Bearer",
        expires_at=now + timedelta(days=30),
        user=UserInfo(id="user_1", email="user@example.com", name="Test User"),
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
# platform_request Tests
# =============================================================================


class TestPlatformRequest:
    """Tests for the platform_request function."""

    @pytest.fixture(autouse=True)
    def _mock_active_workspace(self) -> Any:
        """Provide a default workspace ID so require_workspace=True doesn't fail."""
        with patch(
            "osmosis_ai.platform.auth.platform_client.get_active_workspace_id",
            return_value="default_ws_test",
        ):
            yield

    # -------------------------------------------------------------------------
    # Credential Loading
    # -------------------------------------------------------------------------

    @patch("osmosis_ai.platform.auth.platform_client.load_credentials")
    def test_raises_when_no_credentials_found(self, mock_load: MagicMock) -> None:
        """Verify CLIError when no credentials exist."""
        mock_load.return_value = None

        with pytest.raises(CLIError, match="Not logged in"):
            platform_request("/api/test")

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    @patch("osmosis_ai.platform.auth.platform_client.load_credentials")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    @patch("osmosis_ai.platform.auth.platform_client.load_credentials")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_constructs_correct_url(self, mock_urlopen: MagicMock) -> None:
        """Verify the full URL is built from PLATFORM_URL + endpoint."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        with patch(
            "osmosis_ai.platform.auth.platform_client.PLATFORM_URL",
            "https://test.osmosis.ai",
        ):
            platform_request("/api/v1/verify", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.full_url == "https://test.osmosis.ai/api/v1/verify"

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_sets_required_headers(self, mock_urlopen: MagicMock) -> None:
        """Verify Authorization, Content-Type, and User-Agent headers are set."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials(access_token="my-token")

        platform_request("/api/test", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("Authorization") == "Bearer my-token"
        assert request_obj.get_header("Content-type") == "application/json"
        assert "osmosis-cli/" in request_obj.get_header("User-agent")

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_adds_workspace_header_when_workspace_id_provided(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify X-Osmosis-Org header uses explicit workspace_id over active workspace."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", credentials=creds, workspace_id="ws_123")

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("X-osmosis-org") == "ws_123"

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_adds_workspace_header_from_active_workspace(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify X-Osmosis-Org header uses active workspace when no workspace_id provided."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        # The autouse fixture provides "default_ws_test" as the active workspace
        platform_request("/api/test", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.get_header("X-osmosis-org") == "default_ws_test"

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_no_workspace_header_when_require_workspace_false(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify X-Osmosis-Org header is omitted when require_workspace=False."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", credentials=creds, require_workspace=False)

        request_obj = mock_urlopen.call_args[0][0]
        assert "X-osmosis-org" not in request_obj.headers

    def test_raises_when_require_workspace_true_but_no_workspace(self) -> None:
        """Verify PlatformAPIError when require_workspace=True but no workspace is available."""
        creds = _make_credentials()

        with patch(
            "osmosis_ai.platform.auth.platform_client.get_active_workspace_id",
            return_value=None,
        ):
            with pytest.raises(PlatformAPIError, match="No workspace selected"):
                platform_request("/api/test", credentials=creds)

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_get_request_has_no_body(self, mock_urlopen: MagicMock) -> None:
        """Verify GET requests send no request body."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", method="GET", credentials=creds)

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.data is None
        assert request_obj.get_method() == "GET"

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_timeout_is_passed_to_urlopen(self, mock_urlopen: MagicMock) -> None:
        """Verify the timeout parameter is forwarded to urlopen."""
        mock_urlopen.return_value = _make_http_response({"ok": True})
        creds = _make_credentials()

        platform_request("/api/test", timeout=5.0, credentials=creds)

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 5.0

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.reset_session")
    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_401_triggers_cleanup_and_raises(
        self, mock_urlopen: MagicMock, mock_reset: MagicMock
    ) -> None:
        """Verify 401 response triggers session reset and raises."""
        mock_urlopen.side_effect = _make_http_error(401, "Unauthorized")
        creds = _make_credentials()

        with pytest.raises(AuthenticationExpiredError, match="expired or been revoked"):
            platform_request("/api/test", credentials=creds)

        mock_reset.assert_called_once()

    @patch("osmosis_ai.platform.auth.platform_client.reset_session")
    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_401_with_cleanup_disabled_skips_reset(
        self, mock_urlopen: MagicMock, mock_reset: MagicMock
    ) -> None:
        """Verify 401 with cleanup_on_401=False raises without calling reset_session."""
        mock_urlopen.side_effect = _make_http_error(401, "Unauthorized")
        creds = _make_credentials()

        with pytest.raises(AuthenticationExpiredError, match="expired or been revoked"):
            platform_request("/api/test", credentials=creds, cleanup_on_401=False)

        mock_reset.assert_not_called()

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_http_error_with_empty_body(self, mock_urlopen: MagicMock) -> None:
        """Verify HTTP errors with empty body still produce a useful message."""
        mock_urlopen.side_effect = _make_http_error(502, "")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError) as exc_info:
            platform_request("/api/test", credentials=creds)

        assert "HTTP 502" in str(exc_info.value)
        assert exc_info.value.status_code == 502

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_url_error_raises_platform_api_error(self, mock_urlopen: MagicMock) -> None:
        """Verify URLError is wrapped in PlatformAPIError."""
        mock_urlopen.side_effect = URLError("Name resolution failed")
        creds = _make_credentials()

        with pytest.raises(PlatformAPIError, match="Connection error"):
            platform_request("/api/test", credentials=creds)

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
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

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    @patch("osmosis_ai.platform.auth.platform_client.load_credentials")
    def test_explicit_none_credentials_triggers_load(
        self, mock_load: MagicMock, mock_urlopen: MagicMock
    ) -> None:
        """Verify passing credentials=None explicitly still loads from storage."""
        creds = _make_credentials()
        mock_load.return_value = creds
        mock_urlopen.return_value = _make_http_response({"ok": True})

        platform_request("/api/test", credentials=None)

        mock_load.assert_called_once()

    @patch("osmosis_ai.platform.auth.platform_client.load_credentials")
    def test_load_returns_none_raises_cli_error(self, mock_load: MagicMock) -> None:
        """Verify CLIError when load_credentials returns None."""
        mock_load.return_value = None

        with pytest.raises(CLIError, match="Not logged in"):
            platform_request("/api/test", credentials=None)


# =============================================================================
# revoke_cli_token Tests
# =============================================================================


class TestRevokeCLIToken:
    """Tests for the revoke_cli_token function."""

    def test_returns_false_when_no_token_id(self) -> None:
        """Verify revoke_cli_token returns False when credentials have no token_id."""
        creds = _make_credentials()
        creds.token_id = None
        assert revoke_cli_token(creds) is False

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_returns_true_on_successful_revocation(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Verify revoke_cli_token returns True when the HTTP call succeeds."""
        creds = _make_credentials()
        creds.token_id = "tok_123"
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert revoke_cli_token(creds) is True
        mock_urlopen.assert_called_once()

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_returns_true_on_401(self, mock_urlopen: MagicMock) -> None:
        """A 401 means the token is already gone — that counts as success."""
        creds = _make_credentials()
        creds.token_id = "tok_123"
        mock_urlopen.side_effect = HTTPError(
            url="http://test", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        assert revoke_cli_token(creds) is True

    @patch("osmosis_ai.platform.auth.platform_client.urlopen")
    def test_logs_warning_to_stderr_on_failure(self, mock_urlopen: MagicMock) -> None:
        """Verify revoke_cli_token writes a warning to stderr when revocation fails."""
        creds = _make_credentials()
        creds.token_id = "tok_456"
        mock_urlopen.side_effect = HTTPError(
            url="http://test", code=500, msg="Server Error", hdrs=None, fp=None
        )

        import io
        import sys

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            result = revoke_cli_token(creds)
        finally:
            sys.stderr = old_stderr

        assert result is False
        warning = captured.getvalue()
        assert "Warning: failed to revoke CLI token server-side" in warning
        assert "HTTP 500" in warning
