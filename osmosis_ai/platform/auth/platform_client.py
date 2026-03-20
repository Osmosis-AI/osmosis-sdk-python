"""HTTP client for Osmosis Platform API with automatic 401 handling."""

from __future__ import annotations

import contextlib
import json
import sys
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from osmosis_ai.consts import PACKAGE_VERSION

from .config import PLATFORM_URL
from .credentials import load_credentials
from .local_config import get_active_workspace_id, reset_session

if TYPE_CHECKING:
    from .credentials import Credentials


class AuthenticationExpiredError(Exception):
    """Raised when the stored credentials are invalid, expired, or revoked."""


class PlatformAPIError(Exception):
    """Raised when a Platform API call fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SubscriptionRequiredError(PlatformAPIError):
    """Raised when the workspace requires an active subscription for the requested action."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or "Active subscription required",
            status_code=403,
        )


def _handle_401_and_cleanup() -> None:
    """Handle 401 by clearing credentials and workspace context."""
    reset_session()
    raise AuthenticationExpiredError(
        "Your session has expired or been revoked. "
        "Please run 'osmosis login' to re-authenticate."
    )


def revoke_cli_token(credentials: Credentials) -> bool:
    """Best-effort server-side revocation of a CLI token.

    Returns True if revoked (or already expired/revoked), False on error.
    The caller should still delete local credentials regardless.

    Uses a direct HTTP call instead of ``platform_request`` to avoid its
    automatic 401 handler, which would call ``reset_session()`` and nuke
    local workspace state — an unwanted side-effect during login/logout.
    A 401 here simply means the token is already gone, which is success.
    """
    if not credentials.token_id:
        return False

    url = f"{PLATFORM_URL}/api/cli/tokens/{credentials.token_id}"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
        },
        method="DELETE",
    )

    try:
        with urlopen(request, timeout=5):
            return True
    except HTTPError as e:
        if e.code == 401:
            # Token already expired/revoked — goal achieved.
            return True
        sys.stderr.write(
            f"Warning: failed to revoke CLI token server-side: HTTP {e.code}\n"
        )
        return False
    except (URLError, OSError):
        # Network errors are not critical for best-effort revocation.
        return False


def platform_request(
    endpoint: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    credentials: Credentials | None = None,
    workspace_id: str | None = None,
    require_workspace: bool = True,
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
        workspace_id: Optional workspace ID for X-Osmosis-Org header. If not
            provided and require_workspace is True, uses the active workspace.
        require_workspace: If True, requires a workspace context and adds
            X-Osmosis-Org header. If False, omits workspace context.

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
        "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
    }

    # Add workspace context if required
    if require_workspace:
        resolved_workspace_id = workspace_id or get_active_workspace_id()
        if not resolved_workspace_id:
            raise PlatformAPIError(
                "No active workspace. Run 'osmosis workspace switch' to select one."
            )
        req_headers["X-Osmosis-Org"] = resolved_workspace_id

    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode() if data is not None else None
    request = Request(url, data=body, headers=req_headers, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status == 204:
                return {}
            raw = response.read()
            if not raw:
                return {}
            result: dict[str, Any] = json.loads(raw.decode())
            return result
    except HTTPError as e:
        if e.code == 401:
            _handle_401_and_cleanup()

        # Best-effort capture of structured error message from response body
        detail = ""
        error_body: dict[str, Any] = {}
        try:
            raw = e.read()
            text = raw.decode("utf-8", errors="replace").strip() if raw else ""
            if text:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    parsed = json.loads(text)
                    # Only treat as error_body if it's actually a dict
                    if isinstance(parsed, dict):
                        error_body = parsed
                # Prefer structured error/message field over raw body
                error_msg = error_body.get("error") or error_body.get("message")
                if error_msg and isinstance(error_msg, str):
                    detail = f" {error_msg}"
                elif text:
                    if len(text) > 200:
                        text = text[:200] + "...(truncated)"
                    detail = f" Response: {text}"
        except Exception:
            pass

        # Detect subscription-required responses (403 with subscription message)
        if e.code == 403 and isinstance(error_body, dict):
            error_msg = error_body.get("error", "")
            if isinstance(error_msg, str):
                if "subscription" in error_msg.lower():
                    raise SubscriptionRequiredError(error_msg) from e
                if "workspace" in error_msg.lower() and "access" in error_msg.lower():
                    raise PlatformAPIError(
                        f"{error_msg}\n"
                        "Your workspace context may be stale. "
                        "Run 'osmosis login' or 'osmosis workspace' to re-select.",
                        e.code,
                    ) from e

        raise PlatformAPIError(f"API error: HTTP {e.code}.{detail}", e.code) from e
    except URLError as e:
        raise PlatformAPIError(f"Connection error: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise PlatformAPIError("Invalid JSON response from platform") from e
