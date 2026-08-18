"""Configuration constants for Osmosis authentication."""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from osmosis_ai.cli.errors import CLIError

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
    try:
        working = raw
        head = urlparse(working)
        if not (head.scheme and head.netloc):
            # Dev .env files commonly carry scheme-less URLs like
            # localhost:3000; urlparse reads the host as the scheme there,
            # which used to fail every command downstream. A substring check
            # on "://" would misread a query that embeds a URL.
            probe = urlparse(f"//{working}")
            scheme = "http" if _is_loopback((probe.hostname or "").lower()) else "https"
            working = f"{scheme}://{working}"
        parsed = urlparse(working)
        if not parsed.scheme or not parsed.netloc:
            return raw
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        # The message must stand on its own and the error must stay a CLIError
        # the envelope path can classify.
        raise CLIError(
            f"Invalid platform URL '{raw}'",
            code="VALIDATION",
        ) from None
    # IPv6 literals contain colons and must stay bracketed inside the netloc.
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{host_for_netloc}:{port}"
    else:
        netloc = host_for_netloc

    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def is_insecure_platform_url(platform_url: str) -> bool:
    """True when the URL would send credentials over plaintext to a remote host.

    Surfacing insecure URLs is the CLI's job (``_refuse_insecure_platform_url``
    in ``cli/main.py``): a ``warnings.warn`` here would be suppressed by the
    CLI's warning filter in every supported path.
    """
    parsed = urlparse(platform_url)
    return parsed.scheme.lower() != "https" and not _is_loopback(parsed.hostname or "")


def get_platform_url() -> str:
    """Resolve the active platform URL from the current process environment."""
    return normalize_platform_url(
        os.environ.get("OSMOSIS_PLATFORM_URL", DEFAULT_PLATFORM_URL)
    )


# Configuration directory and credentials file
CONFIG_DIR = Path.home() / ".config" / "osmosis"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
CACHE_DIR = CONFIG_DIR / "cache"

# Credentials file version
CREDENTIALS_VERSION = 2
