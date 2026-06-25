# Dataset contract

> The dataset shape the SDK validates locally before upload. Uploading and managing datasets (`osmosis dataset upload`, overwrite/processing semantics) is covered at [docs.osmosis.ai](https://docs.osmosis.ai/platform/datasets).

Datasets supply prompts (and optional reference answers) for cloud `osmosis eval submit` and `osmosis train submit` runs. The contract below is enforced by the local validator before any bytes hit the platform.

## What the SDK checks

Constants: [../osmosis_ai/platform/cli/constants.py](../osmosis_ai/platform/cli/constants.py). Validation: [../osmosis_ai/platform/cli/dataset.py](../osmosis_ai/platform/cli/dataset.py) (`_validate_file`, `_check_required_columns`).

| Rule | Value | Source |
|------|-------|--------|
| Allowed extensions | `csv`, `jsonl`, `parquet` | `VALID_EXTENSIONS` |
| Required columns | `system_prompt`, `user_prompt` | `REQUIRED_COLUMNS` |
| Minimum rows | `4` | `MIN_ROW_COUNT` |
| Max file size | 5 GB | `MAX_FILE_SIZE` |

## Required columns

Each row must include exactly these column names — `_check_required_columns` does a **case-sensitive** set match (`REQUIRED_COLUMNS - set(columns)`), so `User_Prompt` or `USER_PROMPT` will not satisfy it:

| Column | Type | Description |
|--------|------|-------------|
| `system_prompt` | `str` | System instruction |
| `user_prompt` | `str` | User turn |

`ground_truth` is **not** required by the validator. It is a conventional optional column that graders and tooling read for reference answers — include it when your `Grader` needs it.

## Additional columns

The validator only checks that the required columns are *present*; it never rejects extra columns. Any other columns ride along on the row and are available to downstream code (workflows, graders, platform metadata).

## Examples

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({
    "system_prompt": ["You are a helpful calculator."] * 4,
    "user_prompt": ["What is 2 + 2?", "What is 10 * 5?", "What is 9 - 3?", "What is 8 / 2?"],
    "ground_truth": ["4", "50", "6", "4"],   # optional
})
pq.write_table(table, "data.parquet")
```

```jsonl
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 2 + 2?", "ground_truth": "4"}
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 10 * 5?", "ground_truth": "50"}
```

```csv
system_prompt,user_prompt,ground_truth
You are a helpful calculator.,What is 2 + 2?,4
You are a helpful calculator.,What is 10 * 5?,50
```

> At least `MIN_ROW_COUNT` (4) rows are required. Quote CSV fields containing commas or newlines per [RFC 4180](https://tools.ietf.org/html/rfc4180); prefer Parquet or JSONL for rich text.

## Note: rubric eval input is different

`osmosis eval rubric` does **not** use this columnar contract. It reads a messages-based JSONL file ([../osmosis_ai/eval/rubric/dataset.py](../osmosis_ai/eval/rubric/dataset.py)); see [eval.md](./eval.md).

## See also

- [eval.md](./eval.md) — `evaluate_rubric` API and config validation
- [docs.osmosis.ai/platform/datasets](https://docs.osmosis.ai/platform/datasets) — upload + management
