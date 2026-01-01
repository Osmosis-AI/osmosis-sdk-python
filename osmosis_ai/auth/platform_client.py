"""HTTP client for Osmosis Platform API with automatic 401 handling."""

from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import PLATFORM_URL
from .credentials import delete_workspace_credentials, get_active_workspace, load_credentials

if TYPE_CHECKING:
    from .credentials import WorkspaceCredentials


class AuthenticationExpiredError(Exception):
    """Raised when the stored credentials are invalid, expired, or revoked."""

    pass


class PlatformAPIError(Exception):
    """Raised when a Platform API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _handle_401_and_cleanup() -> None:
    """Handle 401 by deleting current workspace credentials and raising error."""
    active_workspace = get_active_workspace()
    if active_workspace:
        delete_workspace_credentials(active_workspace)

    raise AuthenticationExpiredError(
        "Your session has expired or been revoked. "
        "Please run 'osmosis login' to re-authenticate."
    )


def platform_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    credentials: Optional["WorkspaceCredentials"] = None,
) -> dict[str, Any]:
    """Make an authenticated request to the Osmosis Platform API.

    Automatically handles 401 errors by deleting local credentials.

    Args:
        endpoint: API endpoint (e.g., "/api/cli/verify")
        method: HTTP method
        data: Request body data (will be JSON encoded)
        headers: Additional headers
        timeout: Request timeout in seconds
        credentials: Optional explicit credentials override. If not provided,
            uses the active workspace credentials from local storage.

    Returns:
        Parsed JSON response

    Raises:
        AuthenticationExpiredError: If 401 received (credentials auto-deleted)
        PlatformAPIError: For other API errors
    """
    if credentials is None:
        credentials = load_credentials()
    if credentials is None:
        raise AuthenticationExpiredError(
            "No valid credentials found. Please run 'osmosis login' first."
        )

    url = f"{PLATFORM_URL}{endpoint}"

    req_headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "Content-Type": "application/json",
    }
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode() if data else None
    request = Request(url, data=body, headers=req_headers, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        if e.code == 401:
            _handle_401_and_cleanup()
        # Best-effort capture of response body for debugging (truncate to avoid huge logs)
        detail = ""
        try:
            raw = e.read()
            text = raw.decode("utf-8", errors="replace").strip() if raw else ""
            if text:
                if len(text) > 500:
                    text = text[:500] + "...(truncated)"
                detail = f" Response: {text}"
        except Exception:
            # Ignore body read/decoding failures; keep the original status code context.
            pass

        raise PlatformAPIError(f"API error: HTTP {e.code}.{detail}", e.code)
    except URLError as e:
        raise PlatformAPIError(f"Connection error: {e.reason}")
    except json.JSONDecodeError:
        raise PlatformAPIError("Invalid JSON response from platform")
