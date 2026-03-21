"""Shared constants for platform CLI commands."""

import re

# ── Sentinel values for interactive navigation ────────────────────

BACK = "__back__"
CREATE = "__create__"
LOGOUT_ALL = "__all__"

# ── Interactive list defaults ─────────────────────────────────────

DEFAULT_VISIBLE_CHOICES = 10

# ── Common error messages ─────────────────────────────────────────

MSG_SESSION_EXPIRED = (
    "Your session has expired. Please run 'osmosis auth login' to re-authenticate."
)
MSG_NOT_LOGGED_IN = "Not logged in. Run 'osmosis auth login' first."

# ── Cache ─────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 300

# ── Project name validation ───────────────────────────────────────
# Must match the frontend validation rules.

PROJECT_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?\Z")
PROJECT_NAME_MAX = 64
RESERVED_PROJECT_NAMES = frozenset(
    {
        # Org-level route segments
        "projects",
        "data-sources",
        "tools",
        "reward-functions",
        "llm-judges",
        "rollout-servers",
        "settings",
        # System reserved
        "api",
        "admin",
        "new",
        "project",
    }
)

# ── Dataset validation ────────────────────────────────────────────

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"system_prompt", "user_prompt", "ground_truth"}
