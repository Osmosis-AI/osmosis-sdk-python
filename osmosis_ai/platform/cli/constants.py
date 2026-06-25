"""Shared constants for platform CLI commands."""

# ── Sentinel values for interactive navigation ────────────────────

BACK = "__back__"
CREATE = "__create__"
LOGOUT_ALL = "__all__"

# ── Interactive list defaults ─────────────────────────────────────

DEFAULT_VISIBLE_CHOICES = 10

# ── Cache ─────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 300

# ── Dataset validation ────────────────────────────────────────────

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"system_prompt", "user_prompt"}
MIN_ROW_COUNT = 4
