"""Shared constants for the Osmosis Platform package."""

import os

# ── Pagination ───────────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 50
# Mirrors the server's per-page cap; the API rejects larger limits with a 400.
MAX_PAGE_SIZE = 50
# The logs endpoints allow larger pages than resource lists.
MAX_LOG_PAGE_SIZE = 200

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

# ── Dataset contract (docs/datasets.md) ──────────────────────────

# Extensions the upload validator accepts and the local eval runner can read.
# Extending this set requires a matching reader branch in
# ``eval/local/dataset.py`` (``iter_raw_rows``) and a validator branch in
# ``platform/cli/dataset.py`` (``_validate_file``).
VALID_EXTENSIONS: frozenset[str] = frozenset({"csv", "jsonl", "parquet"})
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
REQUIRED_COLUMNS = {"user_prompt", "ground_truth"}
MIN_ROW_COUNT = 4

# The presence of a metadata column selects metadata mode for the entire
# dataset. Every row must then carry a non-empty JSON object. Users may author
# it as a native object (JSONL/Parquet) or as a JSON-object string (CSV;
# tolerated in JSONL).
METADATA_COLUMN = "metadata"
