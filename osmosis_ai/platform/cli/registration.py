"""Platform registration for RolloutServer (CLI / v2 serve).

This module handles registering the rollout server with Osmosis Platform,
including IP detection and health check verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osmosis_ai.rollout_v2.utils.network import PublicIPDetectionError, detect_public_ip

if TYPE_CHECKING:
    from osmosis_ai.platform.auth.credentials import Credentials

logger: logging.Logger = logging.getLogger(__name__)

_BIND_ALL_HOSTS = frozenset({"0.0.0.0", "", "::", "0:0:0:0:0:0:0:0"})


@dataclass
class RegistrationResult:
    """Result of platform registration."""

    success: bool
    server_id: str | None = None
    status: str = "unknown"
    error: str | None = None
    server_info: dict[str, Any] | None = None

    @property
    def is_healthy(self) -> bool:
        """Check if health check passed."""
        return self.status == "healthy"


def get_report_host(host: str) -> str:
    """Get the host address to report to Platform.

    If the server is bound to all interfaces (IPv4/IPv6 bind-all sentinels), returns the
    detected public IP. Otherwise, returns the provided host.

    Args:
        host: The host the server is bound to.

    Returns:
        The host address to report to Platform.

    Raises:
        PublicIPDetectionError: If host is a bind-all sentinel and IP detection fails.
    """
    if host in _BIND_ALL_HOSTS:
        return detect_public_ip()
    return host


def probe_host(bind_host: str) -> str:
    """Host to use for local readiness probes when the server binds to all interfaces.

    HTTP clients cannot connect to bind-all sentinels; loopback is used instead.
    """
    if bind_host in _BIND_ALL_HOSTS:
        return "127.0.0.1"
    return bind_host


def register_with_platform(
    host: str,
    port: int,
    agent_loop_name: str,
    credentials: Credentials,
    api_key: str | None = None,
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
    from osmosis_ai.platform.auth.platform_client import (
        AuthenticationExpiredError,
        PlatformAPIError,
        platform_request,
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

    registration_data: dict[str, Any] = {
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
            timeout=15.0,
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
                success=True,
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


__all__ = [
    "RegistrationResult",
    "get_report_host",
    "probe_host",
    "register_with_platform",
]
