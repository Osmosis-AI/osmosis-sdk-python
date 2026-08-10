# Dataset contract

> The dataset shape the SDK validates locally before upload. Uploading and managing datasets (`osmosis dataset upload`, overwrite/processing semantics) is covered at [docs.osmosis.ai](https://docs.osmosis.ai/platform/datasets).

Datasets supply prompt/answer pairs or metadata-driven examples for cloud `osmosis eval submit` and `osmosis train submit` runs. The contract below is enforced by the local validator before any bytes hit the platform.

## What the SDK checks

Constants: [../osmosis_ai/platform/cli/constants.py](../osmosis_ai/platform/cli/constants.py). Validation: [../osmosis_ai/platform/cli/dataset.py](../osmosis_ai/platform/cli/dataset.py) (`_validate_file`, `_check_required_columns`).

| Rule | Value | Source |
|------|-------|--------|
| Allowed extensions | `csv`, `jsonl`, `parquet` | `VALID_EXTENSIONS` |
| Schema | Metadata mode or prompt mode | `_check_required_columns` |
| Minimum rows | `4` | `MIN_ROW_COUNT` |
| Max file size | 5 GB | `MAX_FILE_SIZE` |

## Schema modes

A dataset must use one mode throughout:

- **Metadata mode:** the dataset has a `metadata` column, and every row contains a valid, non-empty JSON object in that column. `user_prompt` and `ground_truth` are optional.
- **Prompt mode:** the dataset has no `metadata` column, and every row includes `user_prompt` plus `ground_truth` (`label` is accepted as an alias for `ground_truth`).

`system_prompt` is optional in both modes. Column names are case-sensitive in the SDK validator.

For JSONL, every row must have the same set of top-level fields as the first row. CSV headers and Parquet schemas already define a fixed set of columns.

### Metadata values

Every metadata value must be a non-empty JSON object. Nulls, blank strings, `{}`, nested empty objects, inconsistent value types for the same key across rows, and integers outside the signed 64-bit range are rejected. CSV cells and JSONL strings may contain encoded JSON objects; Parquet accepts struct or JSON-object string columns.

## Additional columns

Additional columns are allowed as long as JSONL rows use the same complete field set.

## Examples

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table(
    {
        "system_prompt": ["You are a helpful calculator."] * 4,
        "user_prompt": [
            "What is 2 + 2?",
            "What is 10 * 5?",
            "What is 9 - 3?",
            "What is 8 / 2?",
        ],
        "ground_truth": ["4", "50", "6", "4"],
    }
)
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

Metadata-mode JSONL can omit all prompt fields:

```jsonl
{"metadata": {"task": "multiply", "left": 15, "right": 23}}
{"metadata": {"task": "simplify_fraction", "numerator": 3, "denominator": 9}}
```

## Note: rubric eval input is different

`osmosis eval rubric` does **not** use this columnar contract. It reads a messages-based JSONL file ([../osmosis_ai/eval/rubric/dataset.py](../osmosis_ai/eval/rubric/dataset.py)); see [eval.md](./eval.md).

## See also

- [eval.md](./eval.md) — `evaluate_rubric` API and config validation
- [docs.osmosis.ai/platform/datasets](https://docs.osmosis.ai/platform/datasets) — upload + management
