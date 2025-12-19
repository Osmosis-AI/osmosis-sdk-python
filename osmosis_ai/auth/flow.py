"""Login flow orchestration for Osmosis CLI authentication."""

from __future__ import annotations

import secrets
import socket
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode

from .config import PLATFORM_URL
from .credentials import WorkspaceCredentials, OrganizationInfo, UserInfo, save_credentials
from .local_server import LocalAuthServer, find_available_port


class LoginError(Exception):
    """Error during login flow."""

    pass


@dataclass
class LoginResult:
    """Result of a successful login."""

    user: UserInfo
    organization: OrganizationInfo
    expires_at: datetime
    revoked_previous_tokens: int = 0


def _generate_state() -> str:
    """Generate a cryptographically secure state parameter."""
    return secrets.token_urlsafe(32)


def _get_device_name() -> str:
    """Get the current device name (hostname).

    Returns:
        The hostname of the current machine.
    """
    try:
        return socket.gethostname()
    except Exception:
        return "Unknown"


def _build_login_url(state: str, port: int) -> str:
    """Build the login URL for the browser.

    Args:
        state: The state parameter for CSRF protection.
        port: The local server port for callback.

    Returns:
        The full login URL.
    """
    device_name = _get_device_name()
    params = {
        "state": state,
        "port": str(port),
        "device_name": device_name,
        "redirect_uri": f"http://localhost:{port}/callback",
    }
    return f"{PLATFORM_URL}/cli-auth?{urlencode(params)}"


def login(
    no_browser: bool = False,
    timeout: float = 300.0,
) -> LoginResult:
    """Execute the login flow.

    Args:
        no_browser: If True, print the URL instead of opening browser.
        timeout: Timeout in seconds for waiting for callback.

    Returns:
        LoginResult with user information.

    Raises:
        LoginError: If login fails.
    """
    # Find an available port
    port = find_available_port()
    if port is None:
        raise LoginError(
            f"No available port found in range 8976-8985. "
            "Please close any applications using these ports."
        )

    # Generate state for CSRF protection
    state = _generate_state()

    # Build the login URL
    login_url = _build_login_url(state, port)

    # Start the local server
    server = LocalAuthServer(port, state)

    try:
        # Always print the URL so users can copy/paste if browser doesn't open
        print("Please open this URL in your browser to log in:")
        print(f"\n{login_url}\n")

        if not no_browser:
            print("Attempting to open browser automatically...")
            if webbrowser.open(login_url):
                print("Browser opened successfully.")
            else:
                print("Could not open browser automatically. Please use the URL above.")

        print("\nWaiting for authentication...")

        # Wait for callback
        token, error = server.wait_for_callback(timeout=timeout)
        revoked_count = server.revoked_count

        if error:
            raise LoginError(f"Authentication failed: {error}")

        if not token:
            raise LoginError("No token received from authentication")

        # Verify token and get user info from platform
        user_info, org_info, expires_at, token_id = _verify_and_get_user_info(token)

        credentials = WorkspaceCredentials(
            access_token=token,
            token_type="Bearer",
            expires_at=expires_at,
            user=user_info,
            organization=org_info,
            created_at=datetime.now(timezone.utc),
            token_id=token_id,
        )

        save_credentials(credentials)

        return LoginResult(
            user=user_info,
            organization=org_info,
            expires_at=expires_at,
            revoked_previous_tokens=revoked_count,
        )

    finally:
        server.server_close()


def _verify_and_get_user_info(token: str) -> tuple[UserInfo, OrganizationInfo, datetime, str | None]:
    """Verify token and get user info from the platform.

    Args:
        token: The access token to verify.

    Returns:
        Tuple of (UserInfo, OrganizationInfo, expiration datetime, token_id).

    Raises:
        LoginError: If verification fails.
    """
    import json
    from datetime import timedelta, timezone
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    verify_url = f"{PLATFORM_URL}/api/cli/verify"

    request = Request(
        verify_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())

            if not data.get("valid"):
                raise LoginError("Token verification failed")

            # Get token_id for revocation
            token_id = data.get("token_id")

            user_data = data.get("user", {})
            user_info = UserInfo(
                id=user_data.get("id", ""),
                email=user_data.get("email", ""),
                name=user_data.get("name"),
            )

            # Parse organization info
            org_data = data.get("organization", {})
            org_info = OrganizationInfo(
                id=org_data.get("id", ""),
                name=org_data.get("name", ""),
                role=org_data.get("role", "member"),
            )

            # Parse expiration - default to 90 days if not provided
            expires_at_str = data.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                if expires_at.tzinfo is None:
                    raise LoginError(
                        "Invalid expires_at from platform: expected timezone-aware ISO8601 timestamp"
                    )
            else:
                expires_at = datetime.now(timezone.utc) + timedelta(days=90)

            return user_info, org_info, expires_at, token_id

    except HTTPError as e:
        if e.code == 401:
            raise LoginError("Invalid or expired token")
        raise LoginError(f"Verification failed: HTTP {e.code}")
    except URLError as e:
        raise LoginError(f"Could not connect to platform: {e.reason}")
    except json.JSONDecodeError:
        raise LoginError("Invalid response from platform")
