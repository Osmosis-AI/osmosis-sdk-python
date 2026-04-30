"""Shared constants for the Osmosis Platform package."""

import os

# ── Pagination ───────────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 50

# ── Common error messages ────────────────────────────────────────

MSG_SESSION_EXPIRED = (
    "Your session has expired. Please run 'osmosis auth login' to re-authenticate."
)
MSG_ENV_TOKEN_INVALID = (
    "The OSMOSIS_TOKEN environment variable is invalid or expired. "
    "Run 'unset OSMOSIS_TOKEN' to use saved credentials or interactive login, "
    "or set OSMOSIS_TOKEN to a valid token."
)
MSG_ENV_TOKEN_EXPIRED = (
    "The OSMOSIS_TOKEN environment variable has expired. "
    "Set OSMOSIS_TOKEN to a new token, or run 'unset OSMOSIS_TOKEN' "
    "to use saved credentials or interactive login."
)
MSG_ENV_TOKEN_REVOKED = (
    "The OSMOSIS_TOKEN environment variable has been revoked. "
    "Set OSMOSIS_TOKEN to a new token, or run 'unset OSMOSIS_TOKEN' "
    "to use saved credentials or interactive login."
)
MSG_NOT_LOGGED_IN = "Not logged in. Run 'osmosis auth login' first."

# ── Inference endpoint ───────────────────────────────────────────

# OpenAI-compatible inference base URL. Override for local/dev inference.
DEFAULT_INFERENCE_URL = "https://inference.osmosis.ai"
INFERENCE_URL = os.environ.get("OSMOSIS_INFERENCE_URL", DEFAULT_INFERENCE_URL)
