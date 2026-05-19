# Dataset format

Datasets supply prompts and reference answers for **`osmosis eval run`**. Each row becomes a short message list (system + user) plus optional extra fields carried on the row dict.

## Supported formats

- **Parquet** (recommended) — compact and fast for large tables
- **JSONL** — one JSON object per line
- **CSV** — header row required

## Required columns

Each row must include (case-insensitive column names):

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Reference answer (used by graders and some tooling) |
| `user_prompt` | `str` | User turn |
| `system_prompt` | `str` | System instruction |

## Example (Parquet)

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "system_prompt": ["You are a helpful calculator.", "You are a helpful calculator."],
    "user_prompt": ["What is 2 + 2?", "What is 10 * 5?"],
    "ground_truth": ["4", "50"],
})
pq.write_table(table, "data.parquet")
```

## Example (JSONL)

```jsonl
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 2 + 2?", "ground_truth": "4"}
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 10 * 5?", "ground_truth": "50"}
```

## Example (CSV)

```csv
system_prompt,user_prompt,ground_truth
You are a helpful calculator.,What is 2 + 2?,4
You are a helpful calculator.,What is 10 * 5?,50
```

> Fields with commas or newlines must be quoted per [RFC 4180](https://tools.ietf.org/html/rfc4180). Prefer Parquet or JSONL for rich text.

## Additional columns

Any other columns are kept on the normalized row dict. Downstream code (workflows, graders, or platform metadata) can read them from that structure.

## Uploading datasets

Upload a local dataset file with:

```bash
osmosis dataset upload data.jsonl
```

Add `--yes` (or `-y`) to skip the interactive confirmation prompt, which is
useful in JSON/plain output modes and CI jobs.

Dataset names are derived from the file name without its extension. If a dataset
with the same name already exists, the upload fails instead of auto-renaming the
new dataset. To replace the existing dataset, pass `--overwrite`:

```bash
osmosis dataset upload data.jsonl --overwrite
```

Overwrite creates a new dataset record and soft-deletes the old one. The platform
may reject overwrites while the existing dataset is still uploading, still
processing, or used by an active training run.

## See also

- [Eval](./eval.md)
