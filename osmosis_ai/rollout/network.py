"""Network utilities for Osmosis remote rollout SDK.

This module provides public IP detection with multi-cloud support (AWS, GCP, Azure,
Lambda Labs, and other cloud providers).

Detection Strategy (ordered by accuracy):
1. Cloud metadata services (parallel) - most accurate, local network, no rate limit
2. External IP services (parallel fallback) - for Lambda Labs and other providers

Example:
    from osmosis_ai.rollout.network import detect_public_ip, PublicIPDetectionError

    try:
        public_ip = detect_public_ip()
        print(f"Detected IP: {public_ip}")
    except PublicIPDetectionError as e:
        print(f"Failed to detect IP: {e}")
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import logging
import re
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Timeout for each cloud metadata HTTP request (local network, should be fast)
# Keep short since 169.254.169.254 is a local link-local address
_METADATA_REQUEST_TIMEOUT_SECONDS = 2.0

# Timeout for external IP detection services (internet requests)
_EXTERNAL_SERVICE_TIMEOUT_SECONDS = 10.0

# Total timeout for parallel cloud metadata detection
# AWS IMDSv2 requires 2 sequential requests (token + IP), so this must be
# at least 2x the per-request timeout plus some buffer
_CLOUD_DETECTION_TIMEOUT_SECONDS = _METADATA_REQUEST_TIMEOUT_SECONDS * 2 + 2.0


# ============================================================================
# Exceptions
# ============================================================================


class PublicIPDetectionError(Exception):
    """Raised when public IP detection fails and no fallback is available."""

    pass


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_ipv4(ip: str) -> bool:
    """Validate that the string is a valid IPv4 address.

    Args:
        ip: String to validate.

    Returns:
        True if valid IPv4 address, False otherwise.

    Example:
        validate_ipv4("192.168.1.1")  # True
        validate_ipv4("256.1.1.1")    # False
        validate_ipv4("not-an-ip")    # False
    """
    try:
        addr = ipaddress.ip_address(ip)
        return addr.version == 4
    except ValueError:
        return False


def is_valid_hostname_or_ip(value: str) -> bool:
    """Validate that the string is a plausible hostname, IPv4 address, or host:port.

    This is a basic format check, not DNS resolution.

    Accepts:
        - IPv4 addresses: 192.168.1.1
        - Hostnames: example.com, sub.example.com, localhost, my-host, my_host, a
        - Host:port: example.com:8080, 192.168.1.1:8080

    Rejects:
        - Empty string
        - Strings with spaces
        - Obvious invalid formats

    Args:
        value: String to validate.

    Returns:
        True if valid hostname or IP address format.

    Example:
        is_valid_hostname_or_ip("192.168.1.1")       # True
        is_valid_hostname_or_ip("example.com")       # True
        is_valid_hostname_or_ip("example.com:8080")  # True
        is_valid_hostname_or_ip("")                  # False
        is_valid_hostname_or_ip("has spaces")        # False
    """
    if not value or len(value) > 260:  # 253 + :port
        return False

    # Strip port if present (e.g., "host:8080" -> "host")
    host = value
    if ":" in value:
        # Could be IPv4:port or hostname:port
        parts = value.rsplit(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            host = parts[0]

    # Check if it's a valid IPv4 address
    if validate_ipv4(host):
        return True

    # Check hostname format (RFC 1123, but allowing underscores for practicality)
    # Allow alphanumeric, hyphens, underscores, and dots
    # Each label should be 1-63 chars, total max 253
    if len(host) > 253:
        return False

    # Pattern allows: alphanumeric, hyphens, underscores, dots
    # Single char hostnames are valid (e.g., "a", "1")
    # Labels can't start or end with hyphen
    hostname_pattern = r'^[A-Za-z0-9_]([A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?(\.[A-Za-z0-9_]([A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?)*$|^[A-Za-z0-9]$'
    return bool(re.match(hostname_pattern, host))


def is_private_ip(ip: str) -> bool:
    """Check if an IP address is a private (RFC 1918) address.

    Args:
        ip: IPv4 address string.

    Returns:
        True if private IP, False if public or invalid.

    Example:
        is_private_ip("192.168.1.1")  # True
        is_private_ip("10.0.0.1")     # True
        is_private_ip("54.123.45.67") # False
    """
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


# ============================================================================
# Cloud Metadata Detection
# ============================================================================


def _get_aws_public_ip() -> Optional[str]:
    """Get public IP from AWS EC2 IMDSv2 metadata service.

    AWS IMDSv2 requires a session token before querying metadata.
    This returns the actual public IP assigned to the instance (not NAT Gateway IP).

    Note: Returns None if the instance has no public IP (e.g., private subnet).

    See: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    """
    try:
        # Step 1: Get session token (IMDSv2 requirement)
        token_resp = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=_METADATA_REQUEST_TIMEOUT_SECONDS,
        )
        token_resp.raise_for_status()
        token = token_resp.text

        # Step 2: Get public IPv4 address
        ip_resp = requests.get(
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=_METADATA_REQUEST_TIMEOUT_SECONDS,
        )
        # AWS returns 404 if instance has no public IP
        if ip_resp.status_code == 404:
            logger.info("AWS metadata: instance has no public IP (private subnet)")
            return None
        ip_resp.raise_for_status()
        ip = ip_resp.text.strip()
        if ip and validate_ipv4(ip):
            logger.info(f"Detected public IP via AWS metadata: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"AWS metadata unavailable: {e}")
    return None


def _get_gcp_public_ip() -> Optional[str]:
    """Get public IP from GCP Compute Engine metadata service.

    GCP requires the 'Metadata-Flavor: Google' header.

    See: https://cloud.google.com/compute/docs/metadata/querying-metadata
    """
    try:
        resp = requests.get(
            "http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
            headers={"Metadata-Flavor": "Google"},
            timeout=_METADATA_REQUEST_TIMEOUT_SECONDS,
        )
        # GCP returns 404 if instance has no external IP
        if resp.status_code == 404:
            logger.info("GCP metadata: instance has no external IP")
            return None
        resp.raise_for_status()
        ip = resp.text.strip()
        if ip and validate_ipv4(ip):
            logger.info(f"Detected public IP via GCP metadata: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"GCP metadata unavailable: {e}")
    return None


def _get_azure_public_ip() -> Optional[str]:
    """Get public IP from Azure Instance Metadata Service (IMDS).

    Note: Only works with Basic SKU public IPs. Standard SKU requires
    querying the load balancer metadata endpoint instead.

    See: https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service
    """
    try:
        resp = requests.get(
            "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/publicIpAddress",
            headers={"Metadata": "true"},
            params={"api-version": "2021-02-01", "format": "text"},
            timeout=_METADATA_REQUEST_TIMEOUT_SECONDS,
        )
        # Azure returns 404 if instance has no public IP
        if resp.status_code == 404:
            logger.info("Azure metadata: instance has no public IP (or Standard SKU)")
            return None
        resp.raise_for_status()
        ip = resp.text.strip()
        if ip and validate_ipv4(ip):
            logger.info(f"Detected public IP via Azure metadata: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"Azure metadata unavailable: {e}")
    return None


def detect_from_cloud_metadata() -> Optional[str]:
    """Try cloud metadata services in parallel.

    Cloud metadata services are:
    - Local network (169.254.169.254), so no rate limit concerns
    - Most accurate: returns actual instance public IP, not NAT Gateway IP
    - Fast: typically 10-50ms response time

    Returns the first successful result, or None if all fail.

    Note: We use short per-request timeouts to ensure threads complete quickly.
    Python's ThreadPoolExecutor.cancel() cannot interrupt running threads,
    so we rely on request timeouts for cleanup.

    Supported cloud providers:
    - AWS EC2 (IMDSv2)
    - GCP Compute Engine
    - Azure VMs (Basic SKU public IPs)

    Returns:
        Public IP address string, or None if detection failed.
    """
    cloud_detectors = [
        (_get_aws_public_ip, "AWS"),
        (_get_gcp_public_ip, "GCP"),
        (_get_azure_public_ip, "Azure"),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for fn, name in cloud_detectors}

        try:
            for future in concurrent.futures.as_completed(
                futures, timeout=_CLOUD_DETECTION_TIMEOUT_SECONDS
            ):
                try:
                    ip = future.result(timeout=0)  # Already completed
                    if ip:
                        logger.debug(f"Cloud metadata detection succeeded: {futures[future]}")
                        # Return immediately; other threads will complete on their
                        # own due to short request timeouts
                        return ip
                except Exception as e:
                    logger.debug(f"Cloud metadata {futures[future]} failed: {e}")
                    continue
        except concurrent.futures.TimeoutError:
            logger.debug("Cloud metadata detection timed out (all providers)")

    return None


# ============================================================================
# External IP Service Detection
# ============================================================================


def detect_from_external_services() -> Optional[str]:
    """Get public IP from external services (works on any cloud including Lambda Labs).

    Queries multiple services in PARALLEL and returns the first successful result.

    WARNING: These services return the IP seen by the external service, which may be:
    - The instance's public IP (correct for Lambda Labs, direct public IP)
    - A NAT Gateway IP (incorrect for VPC environments - use cloud metadata instead)

    This function should only be used as a FALLBACK when cloud metadata is unavailable.

    Note: We rely on per-request timeouts for thread cleanup since Python's
    ThreadPoolExecutor.cancel() cannot interrupt running threads.

    Returns:
        Public IP address string, or None if all services failed.
    """
    # Ordered by reliability and speed
    services = [
        ("checkip.amazonaws.com", "https://checkip.amazonaws.com"),
        ("ipify", "https://api.ipify.org"),
        ("icanhazip", "https://icanhazip.com"),
        ("ifconfig.me", "https://ifconfig.me/ip"),
    ]

    def _query_service(name_url: Tuple[str, str]) -> Optional[str]:
        name, url = name_url
        try:
            resp = requests.get(url, timeout=_EXTERNAL_SERVICE_TIMEOUT_SECONDS)
            resp.raise_for_status()
            ip = resp.text.strip()
            if ip and validate_ipv4(ip):
                return ip
            else:
                logger.debug(f"External service {name} returned invalid IP: {ip!r}")
        except Exception as e:
            logger.debug(f"External service {name} unavailable: {e}")
        return None

    # Query services in parallel for faster response
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(services)) as executor:
        futures = {executor.submit(_query_service, svc): svc[0] for svc in services}

        try:
            for future in concurrent.futures.as_completed(
                futures, timeout=_EXTERNAL_SERVICE_TIMEOUT_SECONDS + 1.0
            ):
                try:
                    ip = future.result(timeout=0)
                    if ip:
                        service_name = futures[future]
                        logger.info(f"Detected public IP via external service ({service_name}): {ip}")
                        # Return immediately; other threads will complete on their
                        # own due to request timeouts
                        return ip
                except Exception as e:
                    logger.debug(f"External service query failed: {e}")
                    continue
        except concurrent.futures.TimeoutError:
            logger.debug("External service detection timed out (all services)")

    return None


# ============================================================================
# Main Public IP Detection
# ============================================================================


def detect_public_ip() -> str:
    """Detect public IP address with multi-cloud support.

    Supports AWS, GCP, Azure, Lambda Labs, and other cloud providers.

    DETECTION PRIORITY (ordered by accuracy):
    1. Cloud metadata services (parallel) - most accurate, local network, no rate limit
    2. External IP services (parallel) - fallback for Lambda Labs, etc.

    WHY THIS ORDER MATTERS:
    - Cloud metadata returns the actual instance public IP (accurate)
    - External services return the NAT Gateway IP in VPC environments (inaccurate)
    - By prioritizing cloud metadata, we avoid returning wrong IP in NAT scenarios

    Note: This function does NOT check environment variables. The caller should
    check for explicit overrides (e.g., OSMOSIS_PUBLIC_HOST) before calling this.

    Returns:
        Public IP address as a string (e.g., "54.123.45.67")

    Raises:
        PublicIPDetectionError: If all detection methods fail.

    Example:
        from osmosis_ai.rollout.network import detect_public_ip, PublicIPDetectionError

        try:
            ip = detect_public_ip()
            print(f"Public IP: {ip}")
        except PublicIPDetectionError as e:
            print(f"Could not detect IP: {e}")
    """
    # Priority 1: Cloud metadata services (parallel, most accurate)
    # These are local network calls, no rate limit, and return actual instance IP
    ip = detect_from_cloud_metadata()
    if ip:
        return ip

    # Priority 2: External IP services (parallel fallback)
    # Used for Lambda Labs and other providers without metadata services
    logger.info("Cloud metadata unavailable, trying external IP services...")
    ip = detect_from_external_services()
    if ip:
        return ip

    # All detection methods failed
    raise PublicIPDetectionError(
        "Failed to detect public IP address. All detection methods failed:\n"
        "  1. Cloud metadata (AWS/GCP/Azure): unavailable or returned no public IP\n"
        "  2. External IP services: all failed or timed out\n"
        "\n"
        "To fix this, provide an explicit IP/hostname to your application."
    )


__all__ = [
    "detect_public_ip",
    "PublicIPDetectionError",
    # Validation helpers
    "validate_ipv4",
    "is_valid_hostname_or_ip",
    "is_private_ip",
]
