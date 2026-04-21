"""Shared constants for the Osmosis Platform package."""

import os

# ── Pagination ───────────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 50

# ── Common error messages ────────────────────────────────────────

MSG_SESSION_EXPIRED = (
    "Your session has expired. Please run 'osmosis auth login' to re-authenticate."
)
MSG_NOT_LOGGED_IN = "Not logged in. Run 'osmosis auth login' first."

# ── Inference endpoint ───────────────────────────────────────────

# OpenAI-compatible inference base URL. Override for local/dev inference.
DEFAULT_INFERENCE_URL = "https://inference.osmosis.ai"
INFERENCE_URL = os.environ.get("OSMOSIS_INFERENCE_URL", DEFAULT_INFERENCE_URL)
