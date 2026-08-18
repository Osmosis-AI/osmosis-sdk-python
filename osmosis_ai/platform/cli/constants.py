"""Shared constants for platform CLI commands."""

# ── Dataset validation ────────────────────────────────────────────

VALID_EXTENSIONS: frozenset[str] = frozenset({"csv", "jsonl", "parquet"})
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"user_prompt", "ground_truth"}
MIN_ROW_COUNT = 4

# The presence of a metadata column selects metadata mode for the entire
# dataset. Every row must then carry a non-empty JSON object. Users may author
# it as a native object (JSONL/Parquet) or as a JSON-object string (CSV;
# tolerated in JSONL).
METADATA_COLUMN = "metadata"
