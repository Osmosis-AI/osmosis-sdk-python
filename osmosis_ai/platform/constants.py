"""Shared constants for the Osmosis Platform package."""

# ── Pagination ───────────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 50

# ── Common error messages ────────────────────────────────────────

MSG_SESSION_EXPIRED = (
    "Your session has expired. Please run 'osmosis auth login' to re-authenticate."
)
MSG_NOT_LOGGED_IN = "Not logged in. Run 'osmosis auth login' first."
