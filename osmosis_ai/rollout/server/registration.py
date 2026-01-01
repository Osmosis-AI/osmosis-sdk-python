"""Platform registration for RolloutServer.

This module handles registering the rollout server with Osmosis Platform,
including IP detection and health check verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from osmosis_ai.rollout.network import detect_public_ip, PublicIPDetectionError

if TYPE_CHECKING:
    from osmosis_ai.auth.credentials import WorkspaceCredentials

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of platform registration."""

    success: bool
    server_id: Optional[str] = None
    status: str = "unknown"
    error: Optional[str] = None
    server_info: Optional[Dict[str, Any]] = None

    @property
    def is_healthy(self) -> bool:
        """Check if health check passed."""
        return self.status == "healthy"


def get_public_ip() -> str:
    """Get the public IP address of this machine.

    Uses multi-cloud detection (AWS/GCP/Azure metadata, external IP services).

    Returns:
        Public IP address string.

    Raises:
        PublicIPDetectionError: If all detection methods fail.
    """
    return detect_public_ip()


def get_report_host(host: str) -> str:
    """Get the host address to report to Platform.

    If the server is bound to 0.0.0.0 (all interfaces), returns the
    detected public IP. Otherwise, returns the provided host.

    Args:
        host: The host the server is bound to.

    Returns:
        The host address to report to Platform.

    Raises:
        PublicIPDetectionError: If host is 0.0.0.0 and IP detection fails.
    """
    if host == "0.0.0.0":
        return get_public_ip()
    return host


def register_with_platform(
    host: str,
    port: int,
    agent_loop_name: str,
    credentials: "WorkspaceCredentials",
    api_key: Optional[str] = None,
) -> RegistrationResult:
    """Register the rollout server with Osmosis Platform.

    Sends a registration request to Platform, which will create a record
    and perform a health check on the server.

    Args:
        host: The host the server is bound to.
        port: The port the server is listening on.
        agent_loop_name: Name of the agent loop being served.
        credentials: Workspace credentials for authenticating the registration
            request to Osmosis Platform (i.e., from `osmosis login`).
        api_key: RolloutServer API key used by TrainGate to authenticate when
            calling this server (sent as `Authorization: Bearer <api_key>`).
            This is NOT related to the `osmosis login` token.

    Returns:
        RegistrationResult with status and any error information.
    """
    from osmosis_ai.auth.platform_client import (
        platform_request,
        PlatformAPIError,
        AuthenticationExpiredError,
    )

    try:
        report_host = get_report_host(host)
    except PublicIPDetectionError as e:
        logger.error("Failed to detect public IP for registration: %s", e)
        return RegistrationResult(
            success=False,
            status="error",
            error="Failed to detect public IP. Please provide an explicit host address.",
        )

    logger.info(
        "Registering with Platform: agent=%s, address=%s:%d",
        agent_loop_name,
        report_host,
        port,
    )

    # Build registration data
    registration_data: Dict[str, Any] = {
        "host": report_host,
        "port": port,
        "agent_loop_name": agent_loop_name,
    }
    if api_key is not None:
        registration_data["api_key"] = api_key

    try:
        result = platform_request(
            "/api/cli/rollout-server/register",
            method="POST",
            data=registration_data,
            timeout=15.0,  # Allow time for health check
            credentials=credentials,
        )

        server_id = result.get("id")
        status = result.get("status", "unknown")
        health_result = result.get("health_check_result", {})

        if status == "healthy":
            return RegistrationResult(
                success=True,
                server_id=server_id,
                status=status,
                server_info=health_result.get("server_info"),
            )
        else:
            return RegistrationResult(
                success=True,  # Registration succeeded, health check failed
                server_id=server_id,
                status=status,
                error=health_result.get("error", "Health check failed"),
            )

    except AuthenticationExpiredError as e:
        logger.error("Authentication expired during registration: %s", e)
        return RegistrationResult(
            success=False,
            status="error",
            error=str(e),
        )

    except PlatformAPIError as e:
        logger.error("Platform API error during registration: %s", e)
        return RegistrationResult(
            success=False,
            status="error",
            error=str(e),
        )

    except Exception as e:
        logger.error("Unexpected error during registration: %s", e, exc_info=True)
        return RegistrationResult(
            success=False,
            status="error",
            error=f"Registration failed: {e}",
        )


def print_registration_result(
    result: RegistrationResult,
    host: str,
    port: int,
    agent_loop_name: str,
    api_key: Optional[str] = None,  # noqa: ARG001
) -> None:
    """Print the registration result to console.

    Args:
        result: The registration result to display.
        host: The host the server is bound to.
        port: The port the server is listening on.
        agent_loop_name: Name of the agent loop being served.
        api_key: API key for this server (unused, kept for compatibility).
    """
    try:
        report_host = get_report_host(host)
    except PublicIPDetectionError:
        report_host = host  # Fallback to original host for display

    if result.is_healthy:
        print(f"\n[OK] Registered with Osmosis Platform")
        print(f"     Agent: {agent_loop_name}")
        print(f"     Address: {report_host}:{port}")
        print(f"     Status: {result.status}")
        if result.server_info:
            active = result.server_info.get("active_rollouts", 0)
            print(f"     Active rollouts: {active}")
    elif result.success:
        # Registration succeeded but health check failed
        print(f"\n[WARNING] Registered but health check failed")
        print(f"     Agent: {agent_loop_name}")
        print(f"     Address: {report_host}:{port}")
        print(f"     Status: {result.status}")
        print(f"     Error: {result.error}")
        print()
        print("     The server will continue running.")
        print(f"     Note: Platform cannot reach http://{report_host}:{port}")
        print("     Tip: Use a VM with public IP and ensure the port is open.")
    else:
        # Registration failed entirely
        print(f"\n[WARNING] Failed to register with Platform")
        print(f"     Error: {result.error}")
        print()
        print("     The server will continue running without registration.")


__all__ = [
    "RegistrationResult",
    "get_public_ip",
    "get_report_host",
    "register_with_platform",
    "print_registration_result",
]
