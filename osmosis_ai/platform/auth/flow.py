"""Authentication flows for Osmosis CLI.

Provides device code flow (RFC 8628) for interactive login and
token verification for CI/headless authentication.
"""

from __future__ import annotations

import contextlib
import json
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from osmosis_ai.cli.console import console
from osmosis_ai.consts import PACKAGE_VERSION

from .config import PLATFORM_URL
from .credentials import Credentials, UserInfo


class LoginError(Exception):
    """Error during login flow."""


@dataclass
class VerifyResult:
    """Result of token verification against the platform."""

    user: UserInfo
    expires_at: datetime
    token_id: str | None


@dataclass
class LoginResult:
    """Result of a successful login."""

    user: UserInfo
    expires_at: datetime

    @classmethod
    def from_verify_result(cls, verified: VerifyResult) -> LoginResult:
        """Build a LoginResult from a VerifyResult."""
        return cls(
            user=verified.user,
            expires_at=verified.expires_at,
        )


@dataclass
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_expires_at(raw: str | None) -> datetime:
    """Parse an ISO 8601 expires_at string into a timezone-aware datetime.

    Falls back to 90 days from now if the value is missing.

    Raises:
        LoginError: If the timestamp is naive (no timezone) or already expired.
    """
    if not raw:
        return datetime.now(UTC) + timedelta(days=90)

    expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if expires_at.tzinfo is None:
        raise LoginError(
            "Invalid expires_at from platform: expected timezone-aware ISO8601 timestamp"
        )
    if datetime.now(UTC) >= expires_at:
        raise LoginError(
            "Received token is already expired. Please check system clock or try again."
        )
    return expires_at


def _get_device_name() -> str:
    """Get the current device name (hostname)."""
    try:
        return socket.gethostname()
    except Exception:
        return "Unknown"


def _copy_to_clipboard(text: str) -> bool:
    """Try to copy text to system clipboard. Returns True on success."""
    system = platform.system()
    try:
        if system == "Darwin":
            cmd = ["pbcopy"]
        elif system == "Linux":
            cmd = ["xclip", "-selection", "clipboard"]
        elif system == "Windows":
            cmd = ["clip"]
        else:
            return False
        subprocess.run(cmd, input=text.encode(), check=True, timeout=3)
        return True
    except Exception:
        return False


def _read_error_detail(e: HTTPError) -> str:
    """Extract error message from an HTTPError JSON response body."""
    try:
        body = json.loads(e.read().decode())
        if isinstance(body, dict):
            return body.get("error", "") or body.get("message", "")
    except Exception:
        pass
    return ""


# User-friendly messages keyed by HTTP status code.
_HTTP_ERROR_MESSAGES: dict[int, str] = {
    401: "Authentication failed.",
    403: "Access denied by the platform.",
    429: "Too many requests. Please wait a few minutes and try again.",
    500: "Osmosis platform encountered an internal error. Please try again later.",
    502: "Osmosis platform is temporarily unavailable. Please try again later.",
    503: "Osmosis platform is temporarily unavailable. Please try again later.",
    504: "Osmosis platform is temporarily unavailable. Please try again later.",
}


def _login_error_from_http(
    e: HTTPError, fallback_prefix: str = "Request failed"
) -> LoginError:
    """Build a LoginError with a user-friendly message from an HTTPError.

    Uses the platform's error detail when it adds meaningful context,
    otherwise falls back to a status-code-specific message or a generic one.
    """
    detail = _read_error_detail(e)
    friendly = _HTTP_ERROR_MESSAGES.get(e.code)

    if detail and friendly:
        return LoginError(f"{friendly} ({detail})")
    if friendly:
        return LoginError(friendly)
    if detail:
        return LoginError(f"{fallback_prefix}: {detail} (HTTP {e.code})")
    return LoginError(f"{fallback_prefix}: HTTP {e.code}")


# ---------------------------------------------------------------------------
# Token verification (--token path)
# ---------------------------------------------------------------------------


def verify_token(token: str) -> VerifyResult:
    """Verify token and get user info from the platform.

    Args:
        token: The access token to verify.

    Returns:
        VerifyResult with user, expiration, and token_id.

    Raises:
        LoginError: If verification fails.
    """
    verify_url = f"{PLATFORM_URL}/api/cli/verify"

    request = Request(
        verify_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())

            token_id = data.get("token_id")

            user_data = data.get("user", {})
            user_info = UserInfo(
                id=user_data.get("id", ""),
                email=user_data.get("email", ""),
                name=user_data.get("name"),
            )

            if not user_info.id or not user_info.email:
                raise LoginError("Server returned incomplete user information")

            expires_at = _parse_expires_at(data.get("expires_at"))

            return VerifyResult(
                user=user_info,
                expires_at=expires_at,
                token_id=token_id,
            )

    except HTTPError as e:
        if e.code == 401:
            raise LoginError("Invalid or expired token") from e
        raise _login_error_from_http(e, "Verification failed") from e
    except URLError as e:
        raise LoginError(f"Could not connect to platform: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise LoginError("Invalid response from platform") from e


# ---------------------------------------------------------------------------
# Device code flow (RFC 8628)
# ---------------------------------------------------------------------------


def request_device_code(device_name: str | None = None) -> DeviceCodeResponse:
    """Request a device code from the platform."""
    url = f"{PLATFORM_URL}/api/cli/device/authorize"
    body = json.dumps(
        {
            "deviceName": device_name or _get_device_name(),
        }
    ).encode()

    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            return DeviceCodeResponse(
                device_code=data["device_code"],
                user_code=data["user_code"],
                verification_uri=data["verification_uri"],
                expires_in=data["expires_in"],
                interval=data["interval"],
            )
    except HTTPError as e:
        raise _login_error_from_http(e, "Failed to request device code") from e
    except URLError as e:
        raise LoginError(f"Could not connect to platform: {e.reason}") from e
    except (json.JSONDecodeError, KeyError) as e:
        raise LoginError("Invalid response from platform") from e


def poll_device_token(
    device_code: str,
    interval: int,
    timeout: float,
    on_poll: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Poll for device authorization completion. Returns token response dict."""
    url = f"{PLATFORM_URL}/api/cli/device/token"
    body = json.dumps({"device_code": device_code}).encode()
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
    }
    deadline = time.monotonic() + timeout
    current_interval = interval

    while time.monotonic() < deadline:
        request = Request(url, data=body, headers=req_headers, method="POST")

        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode())
                return data
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                raise _login_error_from_http(e, "Polling failed") from e
            try:
                error_data = json.loads(e.read().decode())
                error_code = error_data.get("error", "")
            except Exception:
                raise LoginError(f"Polling failed: HTTP {e.code}") from e

            if error_code == "authorization_pending":
                if on_poll:
                    on_poll()
                time.sleep(current_interval)
                continue
            elif error_code == "slow_down":
                current_interval = min(current_interval + 5, 30)
                if on_poll:
                    on_poll()
                time.sleep(current_interval)
                continue
            elif error_code == "expired_token":
                raise LoginError("Device code expired. Please try again.") from e
            elif error_code == "access_denied":
                raise LoginError("Authorization was denied.") from e
            else:
                raise LoginError(
                    f"Polling failed: {error_code or f'HTTP {e.code}'}"
                ) from e
        except URLError as e:
            raise LoginError(f"Could not connect to platform: {e.reason}") from e
        except (json.JSONDecodeError, KeyError) as e:
            raise LoginError("Invalid response from platform") from e

    raise LoginError("Device authorization timed out. Please try again.")


def device_login(timeout: float = 600.0) -> tuple[LoginResult, Credentials]:
    """Execute the device code login flow for headless environments.

    Returns:
        Tuple of (LoginResult, Credentials). Caller is responsible for saving credentials.
    """
    with console.spinner("Requesting device code..."):
        device_code_resp = request_device_code()

    console.print()
    copied = _copy_to_clipboard(device_code_resp.user_code)
    if copied:
        console.print("Your one-time code (copied to clipboard):", style="dim")
    else:
        console.print("Your one-time code:", style="dim")
    console.print()
    console.print(f"  {device_code_resp.user_code}", style="bold cyan")
    console.print()
    expires_minutes = device_code_resp.expires_in // 60
    console.print(f"Code expires in {expires_minutes} minutes.", style="dim")
    console.print()

    verification_url = device_code_resp.verification_uri
    console.print_url(
        "Open this URL in your browser: ", verification_url, style="yellow"
    )

    if sys.stdin.isatty():
        import webbrowser

        with contextlib.suppress(Exception):
            webbrowser.open(verification_url)

    console.print()
    effective_timeout = min(timeout, float(device_code_resp.expires_in))

    with console.spinner("Waiting for authorization..."):
        token_response = poll_device_token(
            device_code=device_code_resp.device_code,
            interval=device_code_resp.interval,
            timeout=effective_timeout,
        )

    if token_response is None:
        raise LoginError("Failed to obtain token response from authorization flow")

    token = token_response.get("token")
    if not token:
        raise LoginError("Server response missing token")

    user_data = token_response.get("user", {})
    user = UserInfo(
        id=user_data.get("id", ""),
        email=user_data.get("email", ""),
        name=user_data.get("name"),
    )

    if not user.id or not user.email:
        raise LoginError("Server returned incomplete user information")

    expires_at = _parse_expires_at(token_response.get("expires_at"))
    token_id = token_response.get("token_id")

    creds = Credentials(
        access_token=token,
        token_type="Bearer",
        expires_at=expires_at,
        created_at=datetime.now(UTC),
        user=user,
        token_id=token_id,
    )
    result = LoginResult(user=user, expires_at=expires_at)

    return result, creds
