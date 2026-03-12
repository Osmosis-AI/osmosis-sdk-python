"""Configuration constants for Osmosis authentication."""

from __future__ import annotations

import os
import warnings
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

# Platform URL - can be overridden via environment variable for local development
DEFAULT_PLATFORM_URL = "https://platform.osmosis.ai"
PLATFORM_URL = os.environ.get("OSMOSIS_PLATFORM_URL", DEFAULT_PLATFORM_URL)


def _is_loopback(hostname: str) -> bool:
    """Check whether *hostname* refers to a loopback interface."""
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


if (
    PLATFORM_URL != DEFAULT_PLATFORM_URL
    and not PLATFORM_URL.startswith("https://")
    and not _is_loopback(urlparse(PLATFORM_URL).hostname or "")
):
    warnings.warn(
        f"OSMOSIS_PLATFORM_URL is not HTTPS ({PLATFORM_URL}). "
        "Tokens will be transmitted in plaintext.",
        stacklevel=2,
    )

# Configuration directory and credentials file
CONFIG_DIR = Path.home() / ".config" / "osmosis"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
CACHE_DIR = CONFIG_DIR / "cache"

# Local server port range for OAuth callback
LOCAL_SERVER_PORT_START = 8976
LOCAL_SERVER_PORT_END = 8985

# Token expiration (for display purposes, actual expiration is set by server)
DEFAULT_TOKEN_EXPIRY_DAYS = 90

# Credentials file version
CREDENTIALS_VERSION = 1
