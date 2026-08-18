"""Shared constants for platform CLI commands."""

# ── Dataset validation ────────────────────────────────────────────

VALID_EXTENSIONS = {"csv", "jsonl", "parquet"}
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"user_prompt", "ground_truth"}
MIN_ROW_COUNT = 4
