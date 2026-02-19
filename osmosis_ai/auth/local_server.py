"""Local HTTP server for handling OAuth callback."""

from __future__ import annotations

import html
import socket
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlparse

from .config import LOCAL_SERVER_PORT_END, LOCAL_SERVER_PORT_START

_TEMPLATES = files("osmosis_ai.auth") / "templates"
_ERROR_PLACEHOLDER = "{{ERROR_MESSAGE}}"
_FALLBACK_SUCCESS_HTML = (
    "<html><body><h1>Login Successful</h1>"
    "<p>You can close this window and return to the CLI.</p></body></html>"
)
_FALLBACK_ERROR_HTML = (
    f"<html><body><h1>Login Failed</h1><p>{_ERROR_PLACEHOLDER}</p></body></html>"
)


@lru_cache(maxsize=1)
def _get_success_html() -> str:
    try:
        return (_TEMPLATES / "success.html").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return _FALLBACK_SUCCESS_HTML


@lru_cache(maxsize=1)
def _get_error_html() -> str:
    try:
        content = (_TEMPLATES / "error.html").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return _FALLBACK_ERROR_HTML
    if _ERROR_PLACEHOLDER not in content:
        return _FALLBACK_ERROR_HTML
    return content


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging."""
        pass

    def do_GET(self) -> None:
        """Handle GET request for OAuth callback."""
        parsed = urlparse(self.path)

        if parsed.path == "/callback":
            self._handle_callback(parsed.query)
        elif parsed.path == "/health":
            self._send_response(200, "OK")
        else:
            self._send_response(404, "Not Found")

    def _handle_callback(self, query_string: str) -> None:
        """Process the OAuth callback with token and state."""
        params = parse_qs(query_string)

        token = params.get("token", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]
        error_description = params.get("error_description", [None])[0]
        revoked_count_str = params.get("revoked_count", ["0"])[0]

        server: LocalAuthServer = self.server  # type: ignore

        if error:
            server.error = error_description or error
            self._send_error_page(error_description or error)
            server._token_event.set()
            return

        if not token or not state:
            server.error = "Missing token or state parameter"
            self._send_error_page("Missing required parameters")
            server._token_event.set()
            return

        # Validate state to prevent CSRF
        if state != server.expected_state:
            server.error = "Invalid state parameter"
            self._send_error_page("Invalid state - possible CSRF attack")
            server._token_event.set()
            return

        # Store the token and revoked count
        server.received_token = token
        try:
            server.revoked_count = int(revoked_count_str) if revoked_count_str else 0
        except ValueError:
            server.revoked_count = 0

        # Signal main thread that token is ready, then wait for verification
        server._token_event.set()

        if not server._verification_event.wait(timeout=30):
            self._send_error_page("Verification timed out")
            return

        if server._verification_result is True:
            self._send_success_page()
        else:
            self._send_error_page(str(server._verification_result))

    def _send_response(self, status: int, message: str) -> None:
        """Send a simple text response."""
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

    def _send_success_page(self) -> None:
        """Send a success HTML page that can be closed."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_get_success_html().encode())

    def _send_error_page(self, error_message: str) -> None:
        """Send an error HTML page."""
        template = _get_error_html()
        escaped_message = html.escape(error_message)
        html_content = template.replace(_ERROR_PLACEHOLDER, escaped_message)
        self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode())


class LocalAuthServer(HTTPServer):
    """Local HTTP server for receiving OAuth callback."""

    def __init__(self, port: int, expected_state: str) -> None:
        """Initialize the server.

        Args:
            port: The port to listen on.
            expected_state: The expected state parameter for CSRF validation.
        """
        super().__init__(("localhost", port), AuthCallbackHandler)
        self.expected_state = expected_state
        self.received_token: str | None = None
        self.error: str | None = None
        self.revoked_count: int = 0
        self._shutdown_event = threading.Event()
        # Synchronization for deferred browser response
        self._token_event = threading.Event()
        self._verification_event = threading.Event()
        self._verification_result: object = None  # True = success, str = error message

    def set_verification_result(self, success: bool, error: str | None = None) -> None:
        """Set the verification result and unblock the callback handler.

        Args:
            success: Whether token verification succeeded.
            error: Error message if verification failed.
        """
        self._verification_result = (
            True if success else (error or "Verification failed")
        )
        self._verification_event.set()

    def wait_for_callback(
        self, timeout: float = 300.0
    ) -> tuple[str | None, str | None]:
        """Wait for the OAuth callback.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (token, error). One will be set, the other None.
        """

        def serve_until_done() -> None:
            while not self._shutdown_event.is_set():
                self.handle_request()

        server_thread = threading.Thread(target=serve_until_done, daemon=True)
        server_thread.start()

        if not self._token_event.wait(timeout=timeout):
            self._shutdown_event.set()
            return None, "Authentication timed out"

        return self.received_token, self.error

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._shutdown_event.set()
        super().shutdown()


def find_available_port() -> int | None:
    """Find an available port in the configured range.

    Returns:
        An available port number, or None if no port is available.
    """
    for port in range(LOCAL_SERVER_PORT_START, LOCAL_SERVER_PORT_END + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    return None
