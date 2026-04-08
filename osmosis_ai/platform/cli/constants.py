"""Shared constants for platform CLI commands."""

import re

# ── Sentinel values for interactive navigation ────────────────────

BACK = "__back__"
CREATE = "__create__"
LOGOUT_ALL = "__all__"

# ── Interactive list defaults ─────────────────────────────────────

DEFAULT_VISIBLE_CHOICES = 10

# ── Cache ─────────────────────────────────────────────────────────

CACHE_TTL_SECONDS = 300

# ── Name validation ───────────────────────────────────────────────
# Generic rules for platform entity names (e.g. workspaces).
# Must match the frontend validation rules.

NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?\Z")
NAME_MAX = 64


def validate_name(name: str, *, label: str = "Name") -> str | None:
    """Validate a name against platform naming rules.

    Returns None if valid, or an error message string if invalid.
    """
    if not name:
        return f"{label} is required."
    if len(name) > NAME_MAX:
        return f"{label} must be {NAME_MAX} characters or less."
    if name != name.lower():
        return f"{label} must be lowercase."
    if not NAME_RE.match(name):
        return (
            f"{label} must contain only lowercase letters, digits, and hyphens, "
            "and cannot start or end with a hyphen."
        )
    return None


# ── Dataset validation ────────────────────────────────────────────

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"system_prompt", "user_prompt", "ground_truth"}
