# Dataset Format

Datasets define the test cases for testing and evaluating your agents. They provide the prompts, expected outputs, and optional metadata used by both [test mode](./test-mode.md) and [eval mode](./eval-mode.md). Datasets are used by both **Local Rollout** and **Remote Rollout** modes.

## Supported Formats

- **Parquet** (recommended) -- columnar format, compact and fast for large datasets
- **JSONL** -- one JSON object per line
- **CSV** -- comma-separated values with a header row

## Required Columns

Each row must contain these columns (case-insensitive):

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Expected output (for reward computation) |
| `user_prompt` | `str` | User message to start the conversation |
| `system_prompt` | `str` | System prompt for the LLM |

## Example Dataset (Parquet)

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

## Example Dataset (JSONL)

```jsonl
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 2 + 2?", "ground_truth": "4"}
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 10 * 5?", "ground_truth": "50"}
```

## Example Dataset (CSV)

```csv
system_prompt,user_prompt,ground_truth
You are a helpful calculator.,What is 2 + 2?,4
You are a helpful calculator.,What is 10 * 5?,50
```

> **Note:** Fields containing commas, newlines, or double-quotes must be enclosed in double-quotes per [RFC 4180](https://tools.ietf.org/html/rfc4180). For datasets where prompts contain rich text, Parquet or JSONL are better choices.

## Additional Columns

Any columns beyond the three required ones are passed to `RolloutRequest.metadata`. This lets you include extra context -- such as difficulty level, category tags, or reference data -- that your agent or reward function can access at runtime.

## See Also

- [Test Mode](./test-mode.md) -- test agents locally with external LLMs
- [Eval Mode](./eval-mode.md) -- evaluate agents with eval functions and pass@k
