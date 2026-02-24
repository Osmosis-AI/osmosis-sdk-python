"""Tests for osmosis_ai.platform.auth.local_server."""

from __future__ import annotations

import http.client
import threading
from unittest.mock import patch

from osmosis_ai.platform.auth.local_server import (
    _ERROR_PLACEHOLDER,
    _FALLBACK_ERROR_HTML,
    _FALLBACK_SUCCESS_HTML,
    LocalAuthServer,
    _get_error_html,
    _get_success_html,
    find_available_port,
)

# ---------------------------------------------------------------------------
# Helper: make HTTP requests to a running LocalAuthServer
# ---------------------------------------------------------------------------


def _http_get(port: int, path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", path)
    return conn.getresponse()


# ---------------------------------------------------------------------------
# _get_success_html / _get_error_html
# ---------------------------------------------------------------------------


class TestGetTemplateHtml:
    def setup_method(self) -> None:
        # Clear lru_cache between tests
        _get_success_html.cache_clear()
        _get_error_html.cache_clear()

    def test_success_html_loads(self) -> None:
        html = _get_success_html()
        assert "<!DOCTYPE html>" in html
        assert "Successful" in html or "successful" in html.lower()

    def test_error_html_loads(self) -> None:
        html = _get_error_html()
        assert "<!DOCTYPE html>" in html
        assert "{{ERROR_MESSAGE}}" in html

    def test_success_html_is_cached(self) -> None:
        a = _get_success_html()
        b = _get_success_html()
        assert a is b

    def test_error_html_is_cached(self) -> None:
        a = _get_error_html()
        b = _get_error_html()
        assert a is b

    def test_success_html_fallback_on_missing_template(self) -> None:
        with patch(
            "osmosis_ai.platform.auth.local_server._TEMPLATES",
            **{
                "__truediv__": lambda self, name: (_ for _ in ()).throw(
                    FileNotFoundError
                )
            },
        ):
            # lru_cache is cleared in setup_method, so this will re-read
            result = _get_success_html()
        assert result == _FALLBACK_SUCCESS_HTML

    def test_error_html_fallback_on_missing_template(self) -> None:
        with patch(
            "osmosis_ai.platform.auth.local_server._TEMPLATES",
            **{
                "__truediv__": lambda self, name: (_ for _ in ()).throw(
                    FileNotFoundError
                )
            },
        ):
            result = _get_error_html()
        assert result == _FALLBACK_ERROR_HTML

    def test_error_html_fallback_on_missing_placeholder(self) -> None:
        """Error template without {{ERROR_MESSAGE}} should fall back."""
        bad_template = "<html><body>No placeholder here</body></html>"
        mock_path = patch(
            "osmosis_ai.platform.auth.local_server._TEMPLATES",
        )
        with mock_path as mock_templates:
            mock_file = mock_templates.__truediv__.return_value
            mock_file.read_text.return_value = bad_template
            result = _get_error_html()
        assert result == _FALLBACK_ERROR_HTML
        assert _ERROR_PLACEHOLDER in result


# ---------------------------------------------------------------------------
# AuthCallbackHandler routing
# ---------------------------------------------------------------------------


class TestAuthCallbackHandlerRouting:
    """Test HTTP routing: /callback, /health, and unknown paths."""

    def _start_server(self, state: str = "test-state") -> LocalAuthServer:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state=state)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

    def test_health_endpoint(self) -> None:
        server = self._start_server()
        try:
            resp = _http_get(server.server_address[1], "/health")
            assert resp.status == 200
            assert resp.read() == b"OK"
        finally:
            server.shutdown()
            server.server_close()

    def test_unknown_path_returns_404(self) -> None:
        server = self._start_server()
        try:
            resp = _http_get(server.server_address[1], "/unknown")
            assert resp.status == 404
        finally:
            server.shutdown()
            server.server_close()


# ---------------------------------------------------------------------------
# AuthCallbackHandler - callback scenarios
# ---------------------------------------------------------------------------


class TestAuthCallbackHandlerCallback:
    def _start_server(self, state: str = "test-state") -> LocalAuthServer:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state=state)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

    def test_callback_with_error_param(self) -> None:
        server = self._start_server()
        try:
            resp = _http_get(
                server.server_address[1],
                "/callback?error=access_denied&error_description=User+denied",
            )
            assert resp.status == 400
            body = resp.read().decode()
            assert "User denied" in body
            # Server should record the error
            server._token_event.wait(timeout=2)
            assert server.error == "User denied"
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_missing_token_or_state(self) -> None:
        server = self._start_server()
        try:
            resp = _http_get(
                server.server_address[1],
                "/callback?token=abc",  # missing state
            )
            assert resp.status == 400
            body = resp.read().decode()
            assert "Missing" in body
            server._token_event.wait(timeout=2)
            assert server.error == "Missing token or state parameter"
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_invalid_state_csrf(self) -> None:
        server = self._start_server(state="correct-state")
        try:
            resp = _http_get(
                server.server_address[1],
                "/callback?token=abc&state=wrong-state",
            )
            assert resp.status == 400
            body = resp.read().decode()
            assert "CSRF" in body
            server._token_event.wait(timeout=2)
            assert server.error == "Invalid state parameter"
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_valid_token_verification_success(self) -> None:
        """Happy path: valid token + state, verification succeeds."""
        state = "good-state"
        server = self._start_server(state=state)
        try:
            # Trigger verification success from another thread after token arrives
            def approve_after_token() -> None:
                server._token_event.wait(timeout=5)
                server.set_verification_result(success=True)

            threading.Thread(target=approve_after_token, daemon=True).start()

            resp = _http_get(
                server.server_address[1],
                f"/callback?token=my-token&state={state}&revoked_count=2",
            )
            assert resp.status == 200
            body = resp.read().decode()
            # Should return the success template
            assert "<!DOCTYPE html>" in body
            assert server.received_token == "my-token"
            assert server.revoked_count == 2
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_valid_token_verification_failure(self) -> None:
        """Valid token + state, but verification fails."""
        state = "good-state"
        server = self._start_server(state=state)
        try:

            def reject_after_token() -> None:
                server._token_event.wait(timeout=5)
                server.set_verification_result(success=False, error="Token expired")

            threading.Thread(target=reject_after_token, daemon=True).start()

            resp = _http_get(
                server.server_address[1],
                f"/callback?token=bad-token&state={state}",
            )
            assert resp.status == 400
            body = resp.read().decode()
            assert "Token expired" in body
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_error_message_is_html_escaped(self) -> None:
        server = self._start_server()
        try:
            resp = _http_get(
                server.server_address[1],
                "/callback?error=xss&error_description=%3Cscript%3Ealert(1)%3C/script%3E",
            )
            body = resp.read().decode()
            assert "<script>" not in body
            assert "&lt;script&gt;" in body
        finally:
            server.shutdown()
            server.server_close()

    def test_callback_invalid_revoked_count_defaults_to_zero(self) -> None:
        state = "s"
        server = self._start_server(state=state)
        try:

            def approve() -> None:
                server._token_event.wait(timeout=5)
                server.set_verification_result(success=True)

            threading.Thread(target=approve, daemon=True).start()

            _http_get(
                server.server_address[1],
                f"/callback?token=t&state={state}&revoked_count=not-a-number",
            )
            assert server.revoked_count == 0
        finally:
            server.shutdown()
            server.server_close()


# ---------------------------------------------------------------------------
# LocalAuthServer
# ---------------------------------------------------------------------------


class TestLocalAuthServer:
    def test_set_verification_result_success(self) -> None:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state="s")
        server.set_verification_result(success=True)
        assert server._verification_result is True
        assert server._verification_event.is_set()
        server.server_close()

    def test_set_verification_result_failure_with_message(self) -> None:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state="s")
        server.set_verification_result(success=False, error="bad token")
        assert server._verification_result == "bad token"
        assert server._verification_event.is_set()
        server.server_close()

    def test_set_verification_result_failure_default_message(self) -> None:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state="s")
        server.set_verification_result(success=False)
        assert server._verification_result == "Verification failed"
        server.server_close()

    def test_wait_for_callback_timeout(self) -> None:
        port = find_available_port()
        assert port is not None
        server = LocalAuthServer(port, expected_state="s")
        token, error = server.wait_for_callback(timeout=0.1)
        assert token is None
        assert error == "Authentication timed out"
        server.server_close()


# ---------------------------------------------------------------------------
# find_available_port
# ---------------------------------------------------------------------------


class TestFindAvailablePort:
    def test_returns_port_in_range(self) -> None:
        port = find_available_port()
        assert port is not None
        assert 8976 <= port <= 8985

    def test_returns_none_when_all_ports_busy(self) -> None:
        with patch("osmosis_ai.platform.auth.local_server.socket.socket") as mock_sock:
            mock_instance = mock_sock.return_value.__enter__.return_value
            mock_instance.bind.side_effect = OSError("Address in use")
            assert find_available_port() is None
