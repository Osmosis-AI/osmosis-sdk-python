"""Authentication flows for Osmosis CLI.

Provides device code flow (RFC 8628) for interactive login and
token verification for CI/headless authentication.
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from osmosis_ai.cli.clipboard import copy_to_clipboard
from osmosis_ai.cli.console import console

from .config import get_platform_url
from .credentials import Credentials, UserInfo
from .platform_client import (
    _read_error_body,
    _response_error_message,
    cli_request_headers,
    connection_error_message,
    is_timeout,
    platform_error_code,
    surface_response_version_signal,
    upgrade_required_message,
)


class LoginError(Exception):
    """Error during login flow."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details


@dataclass
class VerifiedWorkspace:
    """Workspace identity resolved server-side from the X-Osmosis-Git header."""

    id: str | None
    name: str
    role: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedWorkspace:
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            role=data.get("role"),
        )


@dataclass
class VerifyResult:
    """Result of token verification against the platform."""

    user: UserInfo
    expires_at: datetime
    token_id: str | None
    workspace: VerifiedWorkspace | None = None


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
    verification_uri_complete: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_expires_at(raw: Any) -> datetime:
    """Parse an ISO 8601 expires_at string into a timezone-aware datetime.

    Falls back to 90 days from now if the value is missing.

    Raises:
        LoginError: If the value is not a parseable ISO 8601 string, is naive
            (no timezone), or is already expired.
    """
    if not raw:
        return datetime.now(UTC) + timedelta(days=90)
    if not isinstance(raw, str):
        raise LoginError("Invalid response from platform")
    try:
        expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as e:
        raise LoginError("Invalid response from platform") from e
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


def _read_json_object(response: Any) -> dict[str, Any]:
    """Parse a successful response body, rejecting anything but a JSON object."""
    data = json.loads(response.read().decode())
    if not isinstance(data, dict):
        raise LoginError("Invalid response from platform")
    return data


def _parse_user(data: dict[str, Any]) -> UserInfo:
    """Build the caller identity from a login response.

    Raises:
        LoginError: If ``user`` is not an object or lacks an id or email.
    """
    user_data = data.get("user", {})
    if not isinstance(user_data, dict):
        raise LoginError("Invalid response from platform")
    user = UserInfo(
        id=user_data.get("id", ""),
        email=user_data.get("email", ""),
        name=user_data.get("name"),
    )
    if not user.id or not user.email:
        raise LoginError("Server returned incomplete user information")
    return user


# User-friendly messages keyed by HTTP status code.
_HTTP_ERROR_MESSAGES: dict[int, str] = {
    401: "Authentication failed.",
    403: "Access denied by the platform.",
    404: "The platform did not recognize this endpoint. Check the platform URL.",
    429: "Too many requests. Please wait a few minutes and try again.",
    500: "Osmosis platform encountered an internal error. Please try again later.",
    502: "Osmosis platform is temporarily unavailable. Please try again later.",
    503: "Osmosis platform is temporarily unavailable. Please try again later.",
    504: "Osmosis platform is temporarily unavailable. Please try again later.",
}

_CLI_TOKEN_ERROR_MESSAGES: dict[str, str] = {
    "AUTH_HEADER_MISSING": "Token is missing.",
    "TOKEN_MISSING": "Token is missing.",
    "TOKEN_EXPIRED": "Token has expired.",
    "TOKEN_INVALID": "Token is invalid.",
    "TOKEN_REVOKED": "Token has been revoked.",
    "UNKNOWN_AUTH_ERROR": "Authentication failed.",
}


def _login_error_from_http(
    e: HTTPError, fallback_prefix: str = "Request failed"
) -> LoginError:
    """Build a LoginError with a user-friendly message from an HTTPError.

    Uses the platform's error detail when it adds meaningful context,
    otherwise falls back to a status-code-specific message or a generic one.
    """
    if e.code == 426:
        body = _read_error_body(e)
        message, version_signal = upgrade_required_message(body)
        # Carry the parsed version signal so the command layer can attach the
        # same structured ``details`` (status/message) a 426 on a regular API
        # call produces, keeping the UPGRADE_REQUIRED JSON contract consistent
        # across the login handshake and the rest of the CLI.
        return LoginError(
            message,
            code="UPGRADE_REQUIRED",
            status_code=426,
            details=version_signal,
        )

    body = _read_error_body(e)
    detail = _response_error_message(body) or ""
    friendly = _HTTP_ERROR_MESSAGES.get(e.code)
    # A bare error code (the legacy ``error`` field) only means something next
    # to the status-specific text; a server-supplied explanation stands alone.
    error_field = body.get("error")
    bare_code = isinstance(error_field, str) and detail == error_field.strip()

    if friendly and (not detail or bare_code):
        message = f"{friendly} ({detail})" if detail else friendly
    elif detail:
        message = f"{fallback_prefix}: {detail} (HTTP {e.code})"
    else:
        message = f"{fallback_prefix}: HTTP {e.code}"
    # Carry the parsed body so the CLI envelope retains the original response
    # alongside ``platform_code``, exactly as a regular API error does.
    return LoginError(
        message,
        code=platform_error_code(body),
        status_code=e.code,
        details=body or None,
    )


# ---------------------------------------------------------------------------
# Token verification (--token path)
# ---------------------------------------------------------------------------


def verify_token(token: str, *, git_identity: str | None = None) -> VerifyResult:
    """Verify token and get user info from the platform.

    Args:
        token: The access token to verify.
        git_identity: Optional Git repository identity sent as the
            X-Osmosis-Git header so the platform resolves the linked
            workspace. When omitted the result's workspace is None.

    Returns:
        VerifyResult with user, expiration, token_id, and workspace.

    Raises:
        LoginError: If verification fails.
    """
    verify_url = f"{get_platform_url()}/api/cli/verify"

    req_headers = cli_request_headers(token=token)
    if git_identity:
        req_headers["X-Osmosis-Git"] = git_identity

    request = Request(verify_url, headers=req_headers)

    try:
        with urlopen(request, timeout=30) as response:
            surface_response_version_signal(response)
            data = _read_json_object(response)

            token_id = data.get("token_id")
            user_info = _parse_user(data)
            expires_at = _parse_expires_at(data.get("expires_at"))

            workspace_data = data.get("workspace")
            workspace = (
                VerifiedWorkspace.from_dict(workspace_data)
                if isinstance(workspace_data, dict)
                else None
            )

            return VerifyResult(
                user=user_info,
                expires_at=expires_at,
                token_id=token_id,
                workspace=workspace,
            )

    except HTTPError as e:
        if e.code == 401:
            error_body = _read_error_body(e)
            error_code = error_body.get("code")
            if isinstance(error_code, str) and error_code in _CLI_TOKEN_ERROR_MESSAGES:
                raise LoginError(
                    _CLI_TOKEN_ERROR_MESSAGES[error_code],
                    code=error_code,
                    status_code=e.code,
                    details=error_body or None,
                ) from e
            raise LoginError(
                _HTTP_ERROR_MESSAGES[e.code],
                status_code=e.code,
                details=error_body or None,
            ) from e
        raise _login_error_from_http(e, "Verification failed") from e
    except (URLError, TimeoutError) as e:
        raise LoginError(connection_error_message(e)) from e
    except (ValueError, TypeError) as e:
        # JSON decoding or a non-UTF-8 body.
        raise LoginError("Invalid response from platform") from e


# ---------------------------------------------------------------------------
# Device code flow (RFC 8628)
# ---------------------------------------------------------------------------


def request_device_code(device_name: str | None = None) -> DeviceCodeResponse:
    """Request a device code from the platform."""
    url = f"{get_platform_url()}/api/cli/device/authorize"
    body = json.dumps(
        {
            "deviceName": device_name or _get_device_name(),
        }
    ).encode()

    request = Request(
        url,
        data=body,
        headers=cli_request_headers(),
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            surface_response_version_signal(response)
            data = _read_json_object(response)
            return DeviceCodeResponse(
                device_code=data["device_code"],
                user_code=data["user_code"],
                verification_uri=data["verification_uri"],
                expires_in=data["expires_in"],
                interval=data["interval"],
                verification_uri_complete=data.get("verification_uri_complete"),
            )
    except HTTPError as e:
        raise _login_error_from_http(e, "Failed to request device code") from e
    except (URLError, TimeoutError) as e:
        raise LoginError(connection_error_message(e)) from e
    except (ValueError, TypeError, KeyError) as e:
        raise LoginError("Invalid response from platform") from e


def _bounded_sleep(seconds: float, deadline: float) -> None:
    """Sleep for ``seconds`` but never past ``deadline`` (a ``time.monotonic`` value)."""
    time.sleep(max(0.0, min(seconds, deadline - time.monotonic())))


def poll_device_token(
    device_code: str,
    interval: int,
    timeout: float,
    on_poll: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Poll for device authorization completion. Returns token response dict."""
    url = f"{get_platform_url()}/api/cli/device/token"
    body = json.dumps({"device_code": device_code}).encode()
    req_headers = cli_request_headers()
    deadline = time.monotonic() + timeout
    current_interval = interval

    # Neither a poll in flight nor the pause after one may outlive the deadline,
    # so the command ends close to the advertised code expiry.
    while (remaining := deadline - time.monotonic()) > 0:
        request = Request(url, data=body, headers=req_headers, method="POST")

        try:
            with urlopen(request, timeout=min(30.0, remaining)) as response:
                surface_response_version_signal(response)
                return _read_json_object(response)
        except HTTPError as e:
            if e.code in (426, 429, 500, 502, 503, 504):
                raise _login_error_from_http(e, "Polling failed") from e
            error_data = _read_error_body(e)
            error_code = error_data.get("error", "")
            # RFC 8628 puts the machine-readable code in ``error``; carry it so
            # the CLI envelope keeps it under details.platform_code.
            code = error_code if isinstance(error_code, str) and error_code else None

            if error_code == "authorization_pending":
                if on_poll:
                    on_poll()
                _bounded_sleep(current_interval, deadline)
                continue
            elif error_code == "slow_down":
                current_interval = min(current_interval + 5, 30)
                if on_poll:
                    on_poll()
                _bounded_sleep(current_interval, deadline)
                continue
            elif error_code == "expired_token":
                raise LoginError(
                    "Device code expired. Please try again.",
                    code=code,
                    status_code=e.code,
                    details=error_data or None,
                ) from e
            elif error_code == "access_denied":
                raise LoginError(
                    "Authorization was denied.",
                    code=code,
                    status_code=e.code,
                    details=error_data or None,
                ) from e
            else:
                raise LoginError(
                    f"Polling failed: {_response_error_message(error_data) or f'HTTP {e.code}'}",
                    code=code,
                    status_code=e.code,
                    details=error_data or None,
                ) from e
        except (URLError, TimeoutError) as e:
            if is_timeout(e):
                # One stalled poll is not fatal; the deadline bounds the loop.
                _bounded_sleep(current_interval, deadline)
                continue
            raise LoginError(connection_error_message(e)) from e
        except (ValueError, TypeError, KeyError) as e:
            raise LoginError("Invalid response from platform") from e

    raise LoginError("Device authorization timed out. Please try again.")


def device_login(timeout: float = 900.0) -> tuple[LoginResult, Credentials]:
    """Execute the device code login flow for headless environments.

    Returns:
        Tuple of (LoginResult, Credentials). Caller is responsible for saving credentials.
    """
    with console.status("Requesting device code..."):
        device_code_resp = request_device_code()

    console.print()
    copied = copy_to_clipboard(device_code_resp.user_code)
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

    # The complete URI carries the device code so the browser lands on a
    # pre-filled authorize page; older servers may not send it.
    verification_url = (
        device_code_resp.verification_uri_complete or device_code_resp.verification_uri
    )
    console.print_url(
        "Open this URL in your browser: ", verification_url, style="yellow"
    )

    if sys.stdin.isatty():
        import webbrowser

        with contextlib.suppress(Exception):
            webbrowser.open(verification_url)

    console.print()
    effective_timeout = min(timeout, float(device_code_resp.expires_in))

    with console.status("Waiting for authorization..."):
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

    user = _parse_user(token_response)
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
