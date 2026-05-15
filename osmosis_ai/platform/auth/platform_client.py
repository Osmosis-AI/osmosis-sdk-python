"""HTTP client for Osmosis Platform API with automatic 401 handling."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.consts import PACKAGE_VERSION
from osmosis_ai.platform.constants import (
    MSG_ENV_TOKEN_EXPIRED,
    MSG_ENV_TOKEN_INVALID,
    MSG_ENV_TOKEN_REVOKED,
    MSG_NOT_LOGGED_IN,
    MSG_SESSION_EXPIRED,
)

from .config import PLATFORM_URL
from .credentials import load_credentials
from .local_config import reset_session

if TYPE_CHECKING:
    from .credentials import Credentials


class AuthenticationExpiredError(Exception):
    """Raised when the stored credentials are invalid, expired, or revoked."""

    def __init__(self, message: str = MSG_SESSION_EXPIRED, *, code: str | None = None):
        super().__init__(message)
        self.code = code


class PlatformAPIError(Exception):
    """Raised when a Platform API call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        error_code: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.field = field
        self.details = details


class SubscriptionRequiredError(PlatformAPIError):
    """Raised when the workspace requires an active subscription for the requested action."""

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message or "Active subscription required",
            status_code=403,
            error_code=error_code,
            field=field,
            details=details,
        )


def _credentials_match_env_token(credentials: Credentials | None) -> bool:
    env_token = os.environ.get("OSMOSIS_TOKEN")
    return bool(
        env_token and credentials is not None and credentials.access_token == env_token
    )


def _read_error_body(e: HTTPError) -> dict[str, Any]:
    try:
        raw = e.read()
        text = raw.decode("utf-8", errors="replace").strip() if raw else ""
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _auth_error_message(error_code: str | None, *, using_env_token: bool) -> str:
    if using_env_token:
        if error_code == "TOKEN_EXPIRED":
            return MSG_ENV_TOKEN_EXPIRED
        if error_code == "TOKEN_REVOKED":
            return MSG_ENV_TOKEN_REVOKED
        return MSG_ENV_TOKEN_INVALID

    if error_code == "TOKEN_REVOKED":
        return (
            "Your session has been revoked. "
            "Please run 'osmosis auth login' to re-authenticate."
        )
    return MSG_SESSION_EXPIRED


_REPO_SCOPE_ERROR_MESSAGES: dict[str, str] = {
    "GIT_SCOPE_REQUIRED": "This command requires a cloned Osmosis Git repository.",
    "GIT_SCOPE_HEADER_REQUIRED": "This command requires a cloned Osmosis Git repository.",
    "GIT_SCOPE_INVALID": (
        "The Git repository identity is invalid. Run from the Platform-connected GitHub repository."
    ),
    "GIT_SCOPE_HEADER_INVALID": (
        "The Git repository identity is invalid. Run from the Platform-connected GitHub repository."
    ),
    "GIT_REPOSITORY_NOT_CONNECTED": (
        "Platform could not resolve this repository. Clone the repository created by Platform or reconnect it there."
    ),
    "GIT_REPOSITORY_ACCESS_DENIED": (
        "Your account does not have access to the workspace connected to this repository. "
        "Run `osmosis auth login` with the correct account."
    ),
    "GIT_SCOPE_HEADER_ACCESS_DENIED": (
        "Platform could not resolve this repository, or your account does not have access "
        "to its workspace. Clone the Platform-connected repository or run "
        "`osmosis auth login` with the correct account."
    ),
}


def _reject_scope_headers(headers: dict[str, str] | None) -> None:
    if not headers:
        return
    forbidden = {"x-osmosis-org", "x-osmosis-git"}
    supplied = {name.casefold() for name in headers}
    if supplied & forbidden:
        raise PlatformAPIError(
            "Osmosis scope headers are managed by the SDK and cannot be supplied by callers.",
            error_code="SCOPE_HEADER_FORBIDDEN",
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
    git_identity: str | None = None,
    require_git_repo: bool = True,
    cleanup_on_401: bool = True,
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
            uses credentials from local storage.
        git_identity: Explicit Git repository identity for X-Osmosis-Git header
            when require_git_repo is True.
        require_git_repo: If True, requires a Git repository context and adds
            X-Osmosis-Git header. If False, omits repository scope.
        cleanup_on_401: If True (default), a 401 response triggers
            ``reset_session()`` which deletes credentials and local config.
            Set to False for non-critical calls (e.g. post-login validation)
            where a transient 401 should not wipe freshly saved state.

    Returns:
        Parsed JSON response

    Raises:
        AuthenticationExpiredError: If 401 received (credentials auto-deleted
            when cleanup_on_401 is True)
        PlatformAPIError: For other API errors
    """
    _reject_scope_headers(headers)

    if credentials is None:
        credentials = load_credentials()
    if credentials is None:
        raise CLIError(MSG_NOT_LOGGED_IN)
    using_env_token = _credentials_match_env_token(credentials)

    url = f"{PLATFORM_URL}{endpoint}"

    req_headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "Content-Type": "application/json",
        "User-Agent": f"osmosis-cli/{PACKAGE_VERSION}",
    }

    if require_git_repo:
        if not git_identity:
            raise PlatformAPIError(
                "Git-scoped platform requests require an explicit git_identity.",
                error_code="GIT_SCOPE_HEADER_REQUIRED",
            )
        req_headers["X-Osmosis-Git"] = git_identity

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
            error_body = _read_error_body(e)
            error_code = error_body.get("code")
            if not isinstance(error_code, str):
                error_code = None
            if cleanup_on_401 and not using_env_token:
                reset_session()
            raise AuthenticationExpiredError(
                _auth_error_message(error_code, using_env_token=using_env_token),
                code=error_code,
            ) from e

        # Best-effort capture of structured error message from response body
        detail = ""
        error_body: dict[str, Any] = {}
        error_code: str | None = None
        field: str | None = None
        try:
            raw = e.read()
            text = raw.decode("utf-8", errors="replace").strip() if raw else ""
            if text:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    parsed = json.loads(text)
                    # Only treat as error_body if it's actually a dict
                    if isinstance(parsed, dict):
                        error_body = parsed
                        if isinstance(error_body.get("code"), str):
                            error_code = error_body["code"]
                        if isinstance(error_body.get("field"), str):
                            field = error_body["field"]
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

        if error_code in _REPO_SCOPE_ERROR_MESSAGES:
            raise PlatformAPIError(
                _REPO_SCOPE_ERROR_MESSAGES[error_code],
                e.code,
                error_code=error_code,
                field=field,
                details=error_body or None,
            ) from e

        if e.code == 403 and error_code in {
            "SUBSCRIPTION_REQUIRED",
            "BILLING_REQUIRED",
        }:
            error_msg = error_body.get("error") or error_body.get("message")
            raise SubscriptionRequiredError(
                error_msg if isinstance(error_msg, str) else None,
                error_code=error_code,
                field=field,
                details=error_body or None,
            ) from e

        # Detect subscription-required responses (403 with subscription message)
        if e.code == 403 and isinstance(error_body, dict):
            error_msg = error_body.get("error", "")
            if isinstance(error_msg, str) and "subscription" in error_msg.lower():
                raise SubscriptionRequiredError(
                    error_msg,
                    error_code=error_code,
                    field=field,
                    details=error_body or None,
                ) from e

        raise PlatformAPIError(
            f"API error: HTTP {e.code}.{detail}",
            e.code,
            error_code=error_code,
            field=field,
            details=error_body or None,
        ) from e
    except URLError as e:
        raise PlatformAPIError(f"Connection error: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise PlatformAPIError("Invalid JSON response from platform") from e
