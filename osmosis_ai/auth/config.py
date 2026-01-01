"""Configuration constants for Osmosis authentication."""

from __future__ import annotations

import os
from pathlib import Path

# Platform URL - can be overridden via environment variable for local development
DEFAULT_PLATFORM_URL = "https://platform.osmosis.ai"
PLATFORM_URL = os.environ.get("OSMOSIS_PLATFORM_URL", DEFAULT_PLATFORM_URL)

# Configuration directory and credentials file
CONFIG_DIR = Path.home() / ".config" / "osmosis"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"

# Local server port range for OAuth callback
LOCAL_SERVER_PORT_START = 8976
LOCAL_SERVER_PORT_END = 8985

# Token expiration (for display purposes, actual expiration is set by server)
DEFAULT_TOKEN_EXPIRY_DAYS = 90

# Credentials file version
CREDENTIALS_VERSION = 1
