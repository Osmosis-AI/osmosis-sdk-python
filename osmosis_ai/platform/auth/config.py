"""Configuration constants for Osmosis authentication."""

from __future__ import annotations

import os
import warnings
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# Platform URL - can be overridden via environment variable for local development
DEFAULT_PLATFORM_URL = "https://platform.osmosis.ai"


def _is_loopback(hostname: str) -> bool:
    """Check whether *hostname* refers to a loopback interface."""
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def normalize_platform_url(url: str | None) -> str:
    """Return the canonical platform base URL used for requests and storage keys."""
    raw = (url or DEFAULT_PLATFORM_URL).strip() or DEFAULT_PLATFORM_URL
    raw = raw.rstrip("/")
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def _warn_if_insecure_platform_url(platform_url: str) -> None:
    parsed = urlparse(platform_url)
    if (
        platform_url != DEFAULT_PLATFORM_URL
        and parsed.scheme.lower() != "https"
        and not _is_loopback(parsed.hostname or "")
    ):
        warnings.warn(
            f"OSMOSIS_PLATFORM_URL is not HTTPS ({platform_url}). "
            "Tokens will be transmitted in plaintext.",
            stacklevel=2,
        )


def get_platform_url() -> str:
    """Resolve the active platform URL from the current process environment."""
    platform_url = normalize_platform_url(
        os.environ.get("OSMOSIS_PLATFORM_URL", DEFAULT_PLATFORM_URL)
    )
    _warn_if_insecure_platform_url(platform_url)
    return platform_url


PLATFORM_URL = get_platform_url()

# Configuration directory and credentials file
CONFIG_DIR = Path.home() / ".config" / "osmosis"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
CACHE_DIR = CONFIG_DIR / "cache"

# Token expiration (for display purposes, actual expiration is set by server)
DEFAULT_TOKEN_EXPIRY_DAYS = 90

# Credentials file version
CREDENTIALS_VERSION = 2
