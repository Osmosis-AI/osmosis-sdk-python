"""Local HTTP server for handling OAuth callback."""

from __future__ import annotations

import html
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .config import LOCAL_SERVER_PORT_END, LOCAL_SERVER_PORT_START


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
            return

        if not token or not state:
            server.error = "Missing token or state parameter"
            self._send_error_page("Missing required parameters")
            return

        # Validate state to prevent CSRF
        if state != server.expected_state:
            server.error = "Invalid state parameter"
            self._send_error_page("Invalid state - possible CSRF attack")
            return

        # Store the token and revoked count
        server.received_token = token
        try:
            server.revoked_count = int(revoked_count_str) if revoked_count_str else 0
        except ValueError:
            server.revoked_count = 0
        self._send_success_page()

    def _send_response(self, status: int, message: str) -> None:
        """Send a simple text response."""
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode())

    def _send_success_page(self) -> None:
        """Send a success HTML page that can be closed."""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Osmosis - Login Successful</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --background: #fafafa;
            --foreground: #18181b;
            --card: #ffffff;
            --card-foreground: #18181b;
            --muted-foreground: #71717a;
            --border: #e4e4e7;
            --primary: #6366f1;
            --primary-light: #818cf8;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --background: #0a0a0b;
                --foreground: #fafafa;
                --card: #18181b;
                --card-foreground: #fafafa;
                --muted-foreground: #a1a1aa;
                --border: #27272a;
                --primary: #818cf8;
                --primary-light: #a5b4fc;
            }
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 16px;
            background: var(--background);
            color: var(--foreground);
        }
        .card {
            width: 100%;
            max-width: 400px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .card-header {
            padding: 24px 24px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 18px;
            font-weight: 600;
            color: var(--foreground);
        }
        .logo svg {
            width: 24px;
            height: 24px;
        }
        .card-content {
            padding: 20px 24px 24px;
            text-align: center;
        }
        h1 {
            font-size: 20px;
            font-weight: 500;
            color: var(--card-foreground);
            margin-bottom: 8px;
        }
        p {
            font-size: 14px;
            color: var(--muted-foreground);
            line-height: 1.5;
        }
        .footer {
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
            text-align: center;
        }
        .footer p {
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="card-header">
            <div class="logo">
                <svg viewBox="0 0 141 141" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <g transform="translate(12, 0)">
                        <path d="M83.9715 50.7784C83.9779 50.4555 83.9936 50.1348 84.0185 49.8167C84.0421 49.2477 84.0539 48.5845 84 48C83.5273 42.8752 79.9618 38.7926 77.2289 36.694C74.3479 34.4816 65.7091 32.691 63.3331 32.6439C61.8442 32.7005 60.3174 32.5506 58.7886 32.1722C50.0879 30.0186 44.618 21.2824 46.5714 12.6592C48.5248 4.03604 57.1617 -1.20864 65.8624 0.94489C72.7969 2.66126 77.6791 8.55903 78.3866 15.2484C79.3072 19.761 81.5386 26.1302 84.9117 29.0113C87.7927 31.472 91.7667 32.9905 95.7141 32.7851C98.0667 32.262 100.587 32.2639 103.112 32.8889C111.812 35.0424 116.62 42.5 116.62 51.1054V87.5984C116.62 96.6149 109.311 103.924 100.294 103.924C91.2777 103.924 83.9683 96.6149 83.9683 87.5984V51.1054C83.9633 51.037 83.9655 50.9244 83.9715 50.7784Z" fill="currentColor"/>
                        <path d="M32.6456 90.1458C32.6393 90.4687 32.6236 90.7894 32.5987 91.1075C32.5751 91.6765 32.5633 92.3397 32.6172 92.9242C33.0899 98.049 36.6554 102.132 39.3883 104.23C42.2693 106.443 50.9081 108.233 53.2841 108.28C54.773 108.224 56.2998 108.374 57.8286 108.752C66.5293 110.906 71.9991 119.642 70.0458 128.265C68.0924 136.888 59.4555 142.133 50.7548 139.979C43.8203 138.263 38.9381 132.365 38.2306 125.676C37.31 121.163 35.0786 114.794 31.7055 111.913C28.8245 109.452 24.8505 107.934 20.9031 108.139C18.5505 108.662 16.0307 108.66 13.5055 108.035C4.80471 105.882 -0.00277342 98.4242 -0.00277267 89.8188L-0.00276948 53.3258C-0.00276869 44.3093 7.30655 37 16.323 37C25.3395 37 32.6489 44.3093 32.6489 53.3258L32.6489 89.8188C32.6539 89.8871 32.6517 89.9998 32.6456 90.1458Z" fill="currentColor"/>
                    </g>
                </svg>
                Osmosis
            </div>
        </div>
        <div class="card-content">
            <h1>Login Successful</h1>
            <p>You have been authenticated successfully. You can close this window and return to the terminal.</p>
            <div class="footer">
                <p>Osmosis CLI</p>
            </div>
        </div>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode())

    def _send_error_page(self, error_message: str) -> None:
        """Send an error HTML page."""
        escaped_message = html.escape(error_message)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Osmosis - Login Failed</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --background: #fafafa;
            --foreground: #18181b;
            --card: #ffffff;
            --card-foreground: #18181b;
            --muted-foreground: #71717a;
            --border: #e4e4e7;
            --primary: #6366f1;
            --destructive: #ef4444;
            --destructive-bg: #fef2f2;
            --destructive-border: #fecaca;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --background: #0a0a0b;
                --foreground: #fafafa;
                --card: #18181b;
                --card-foreground: #fafafa;
                --muted-foreground: #a1a1aa;
                --border: #27272a;
                --primary: #818cf8;
                --destructive: #f87171;
                --destructive-bg: rgba(127, 29, 29, 0.5);
                --destructive-border: #7f1d1d;
            }}
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 16px;
            background: var(--background);
            color: var(--foreground);
        }}
        .card {{
            width: 100%;
            max-width: 400px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .card-header {{
            padding: 24px 24px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
        }}
        .logo {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 18px;
            font-weight: 600;
            color: var(--foreground);
        }}
        .logo svg {{
            width: 24px;
            height: 24px;
        }}
        .card-content {{
            padding: 20px 24px 24px;
            text-align: center;
        }}
        h1 {{
            font-size: 20px;
            font-weight: 500;
            color: var(--card-foreground);
            margin-bottom: 8px;
        }}
        p {{
            font-size: 14px;
            color: var(--muted-foreground);
            line-height: 1.5;
        }}
        .error-box {{
            margin-top: 16px;
            padding: 12px 16px;
            background: var(--destructive-bg);
            border: 1px solid var(--destructive-border);
            border-radius: 8px;
            text-align: left;
        }}
        .error-box p {{
            color: var(--destructive);
            font-size: 13px;
            word-break: break-word;
        }}
        .footer {{
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
            text-align: center;
        }}
        .footer p {{
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="card-header">
            <div class="logo">
                <svg viewBox="0 0 141 141" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <g transform="translate(12, 0)">
                        <path d="M83.9715 50.7784C83.9779 50.4555 83.9936 50.1348 84.0185 49.8167C84.0421 49.2477 84.0539 48.5845 84 48C83.5273 42.8752 79.9618 38.7926 77.2289 36.694C74.3479 34.4816 65.7091 32.691 63.3331 32.6439C61.8442 32.7005 60.3174 32.5506 58.7886 32.1722C50.0879 30.0186 44.618 21.2824 46.5714 12.6592C48.5248 4.03604 57.1617 -1.20864 65.8624 0.94489C72.7969 2.66126 77.6791 8.55903 78.3866 15.2484C79.3072 19.761 81.5386 26.1302 84.9117 29.0113C87.7927 31.472 91.7667 32.9905 95.7141 32.7851C98.0667 32.262 100.587 32.2639 103.112 32.8889C111.812 35.0424 116.62 42.5 116.62 51.1054V87.5984C116.62 96.6149 109.311 103.924 100.294 103.924C91.2777 103.924 83.9683 96.6149 83.9683 87.5984V51.1054C83.9633 51.037 83.9655 50.9244 83.9715 50.7784Z" fill="currentColor"/>
                        <path d="M32.6456 90.1458C32.6393 90.4687 32.6236 90.7894 32.5987 91.1075C32.5751 91.6765 32.5633 92.3397 32.6172 92.9242C33.0899 98.049 36.6554 102.132 39.3883 104.23C42.2693 106.443 50.9081 108.233 53.2841 108.28C54.773 108.224 56.2998 108.374 57.8286 108.752C66.5293 110.906 71.9991 119.642 70.0458 128.265C68.0924 136.888 59.4555 142.133 50.7548 139.979C43.8203 138.263 38.9381 132.365 38.2306 125.676C37.31 121.163 35.0786 114.794 31.7055 111.913C28.8245 109.452 24.8505 107.934 20.9031 108.139C18.5505 108.662 16.0307 108.66 13.5055 108.035C4.80471 105.882 -0.00277342 98.4242 -0.00277267 89.8188L-0.00276948 53.3258C-0.00276869 44.3093 7.30655 37 16.323 37C25.3395 37 32.6489 44.3093 32.6489 53.3258L32.6489 89.8188C32.6539 89.8871 32.6517 89.9998 32.6456 90.1458Z" fill="currentColor"/>
                    </g>
                </svg>
                Osmosis
            </div>
        </div>
        <div class="card-content">
            <h1>Login Failed</h1>
            <p>Something went wrong during authentication.</p>
            <div class="error-box">
                <p>{escaped_message}</p>
            </div>
            <div class="footer">
                <p>Osmosis CLI</p>
            </div>
        </div>
    </div>
</body>
</html>"""
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
        self.received_token: Optional[str] = None
        self.error: Optional[str] = None
        self.revoked_count: int = 0
        self._shutdown_event = threading.Event()

    def wait_for_callback(self, timeout: float = 300.0) -> tuple[Optional[str], Optional[str]]:
        """Wait for the OAuth callback.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (token, error). One will be set, the other None.
        """

        def serve_until_done() -> None:
            while not self._shutdown_event.is_set():
                self.handle_request()
                if self.received_token is not None or self.error is not None:
                    break

        server_thread = threading.Thread(target=serve_until_done)
        server_thread.daemon = True
        server_thread.start()

        server_thread.join(timeout=timeout)

        if server_thread.is_alive():
            self._shutdown_event.set()
            return None, "Authentication timed out"

        return self.received_token, self.error

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._shutdown_event.set()
        super().shutdown()


def find_available_port() -> Optional[int]:
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
