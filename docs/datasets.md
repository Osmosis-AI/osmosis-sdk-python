# Dataset format

Datasets supply prompts and reference answers for **`osmosis rollout test`** and **`osmosis eval run`**. Each row becomes a short message list (system + user) plus optional extra fields carried on the row dict.

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

## See also

- [Test mode](./test-mode.md)
- [Eval mode](./eval-mode.md)
