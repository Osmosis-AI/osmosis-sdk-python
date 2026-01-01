"""API Key generation and validation for RolloutServer.

This module provides utilities for generating secure API keys
and validating incoming requests against them.
"""

from __future__ import annotations

import secrets
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# API Key prefix for identification
API_KEY_PREFIX = "osm_rollout_"

# Length of the random part (32 bytes = 256 bits of entropy)
API_KEY_RANDOM_BYTES = 32


def generate_api_key() -> str:
    """Generate a secure API key for the RolloutServer.

    The key is URL-safe and has the format: osm_rollout_<random>

    Returns:
        A secure, URL-safe API key string.

    Example:
        >>> key = generate_api_key()
        >>> key.startswith("osm_rollout_")
        True
    """
    random_part = secrets.token_urlsafe(API_KEY_RANDOM_BYTES)
    return f"{API_KEY_PREFIX}{random_part}"


def validate_api_key(provided_key: Optional[str], expected_key: str) -> bool:
    """Validate a provided API key against the expected key.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        provided_key: The API key provided in the request.
        expected_key: The expected (server-generated) API key.

    Returns:
        True if the keys match, False otherwise.
    """
    if provided_key is None:
        return False
    return secrets.compare_digest(provided_key, expected_key)


__all__ = [
    "generate_api_key",
    "validate_api_key",
    "API_KEY_PREFIX",
]
