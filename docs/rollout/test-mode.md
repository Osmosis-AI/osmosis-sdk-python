# Test Mode

Test your `RolloutAgentLoop` implementations locally without TrainGate using external LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers).

## Overview

Test mode enables you to:

- Validate agent logic before deploying to training infrastructure
- Use 100+ LLM providers (OpenAI, Anthropic, Groq, Ollama, etc.)
- Run batch tests against datasets with detailed metrics
- Debug step-by-step with interactive mode
- Track token usage, latency, and reward

## Quick Start

### CLI Usage

```bash
# Basic batch test with OpenAI
osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --model gpt-4o

# Use Anthropic Claude
osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --model anthropic/claude-sonnet-4-20250514

# Interactive debugging
osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --interactive

# Start at specific row
osmosis test --agent my_agent:MyAgentLoop --dataset data.jsonl --interactive --row 5
```

### Programmatic Usage

```python
from osmosis_ai.rollout.eval.common import DatasetReader
from osmosis_ai.rollout.eval.common import ExternalLLMClient
from osmosis_ai.rollout.eval.test_mode import LocalTestRunner

# Load dataset
reader = DatasetReader("./test_data.jsonl")
rows = reader.read(limit=10)

# Initialize LLM client
client = ExternalLLMClient("gpt-4o")  # or "anthropic/claude-sonnet-4-20250514"

# Create runner
runner = LocalTestRunner(
    agent_loop=MyAgentLoop(),
    llm_client=client,
)

# Run batch tests
async with client:
    results = await runner.run_batch(
        rows=rows,
        max_turns=10,
        completion_params={"temperature": 0.7},
    )

print(f"Passed: {results.passed}/{results.total}")
print(f"Total tokens: {results.total_tokens}")
```

---

## Dataset Format

Test mode reads datasets in JSON, JSONL, or Parquet format. Each row must contain these columns (case-insensitive):

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Expected output (for reward computation) |
| `user_prompt` | `str` | User message to start the conversation |
| `system_prompt` | `str` | System prompt for the LLM |

Additional columns are passed to `RolloutRequest.metadata`.

### Example Dataset (JSONL)

```jsonl
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 2 + 2?", "ground_truth": "4"}
{"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 10 * 5?", "ground_truth": "50"}
```

### Example Dataset (JSON)

```json
[
  {"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 2 + 2?", "ground_truth": "4"},
  {"system_prompt": "You are a helpful calculator.", "user_prompt": "What is 10 * 5?", "ground_truth": "50"}
]
```

---

## CLI Reference

```
osmosis test [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `-m, --module, --agent MODULE` | Module path to agent loop (format: `module:attribute`) |
| `-d, --dataset FILE` | Path to dataset file (.json, .jsonl, .parquet) |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model MODEL` | `gpt-4o` | Model name (see [Model Format](#model-format)) |
| `--api-key KEY` | env var | API key for the LLM provider |
| `--base-url URL` | - | Base URL for OpenAI-compatible APIs |

### Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-turns N` | `10` | Maximum agent turns per row |
| `--temperature FLOAT` | - | LLM sampling temperature |
| `--max-tokens N` | - | Maximum tokens per completion |
| `--limit N` | all | Maximum rows to test |
| `--offset N` | `0` | Number of rows to skip |

### Mode Options

| Option | Description |
|--------|-------------|
| `-i, --interactive` | Enable interactive step-by-step mode |
| `--row N` | Initial row for interactive mode (requires `--interactive`) |

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Write results to JSON file |
| `-q, --quiet` | Suppress progress output |
| `--debug` | Enable debug output |

### Model Format

Models can be specified in two formats:

- **Simple**: `gpt-4o` (auto-prefixed to `openai/gpt-4o`)
- **LiteLLM format**: `provider/model` (e.g., `anthropic/claude-sonnet-4-20250514`)

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for supported providers.

### Examples

```bash
# Test with GPT-4o (default)
osmosis test -m my_agent:agent_loop -d data.jsonl

# Test with Claude
osmosis test -m my_agent:agent_loop -d data.jsonl --model anthropic/claude-sonnet-4-20250514

# Test with local Ollama
osmosis test -m my_agent:agent_loop -d data.jsonl \
    --model ollama/llama2 \
    --base-url http://localhost:11434

# Test subset of data
osmosis test -m my_agent:agent_loop -d data.jsonl --limit 10 --offset 50

# Save results to file
osmosis test -m my_agent:agent_loop -d data.jsonl -o results.json

# Interactive debugging starting at row 5
osmosis test -m my_agent:agent_loop -d data.jsonl --interactive --row 5
```

---

## Interactive Mode

Interactive mode allows step-by-step debugging of agent execution. After each LLM call, execution pauses to let you inspect the state.

### Commands

| Command | Description |
|---------|-------------|
| `n` / `next` | Execute next LLM call |
| `c` / `continue` | Continue to completion (disable stepping) |
| `m` / `messages` | Show current message history |
| `t` / `tools` | Show available tools |
| `q` / `quit` | Exit interactive session |
| `r` / `row N` | Jump to row N |
| `?` / `help` | Show help |

### Example Session

```
osmosis-rollout-test v0.1.0 (Interactive Mode)

Loading agent: my_agent:agent_loop
  Agent name: calculator
Loading dataset: data.jsonl
  Total rows: 100
Initializing provider: openai
  Model: openai/gpt-4o

[Interactive Mode] Row 0/100
System: You are a helpful calculator.
User: What is 2 + 2?

[Step 1] Waiting for LLM response...
> n

Assistant called tool: calculate(operation="add", a=2, b=2)
Tool result: 4

[Step 2] Waiting for LLM response...
> m

Messages:
  [0] system: You are a helpful calculator.
  [1] user: What is 2 + 2?
  [2] assistant: [tool_call: calculate]
  [3] tool: 4

> c

Continuing without stepping...
Result: COMPLETED (reward=1.0)

[Row 0 Complete] Next row? (n=next, q=quit, r N=jump to row N)
> q
```

---

## API Reference

### DatasetReader

Read and validate test datasets.

```python
from osmosis_ai.rollout.eval.common import DatasetReader

reader = DatasetReader("./data.jsonl")

# Get total row count
total = len(reader)

# Read all rows
rows = reader.read()

# Read with pagination
rows = reader.read(limit=10, offset=20)

# Iterate rows (memory efficient)
for row in reader.iter_rows():
    print(row["user_prompt"])
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to dataset file |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `read(limit, offset)` | `List[DatasetRow]` | Read rows with optional pagination |
| `iter_rows()` | `Iterator[DatasetRow]` | Memory-efficient iterator |
| `__len__()` | `int` | Total row count |

---

### DatasetRow

A single row from the dataset, represented as a `TypedDict`.

```python
class DatasetRow(TypedDict):
    ground_truth: str      # Expected output
    user_prompt: str       # User message
    system_prompt: str     # System prompt
    # Additional columns from the dataset are included as extra dict keys
```

Access fields with dict syntax: `row["user_prompt"]`, `row["ground_truth"]`, etc.

---

### ExternalLLMClient

Call external LLM APIs via LiteLLM.

```python
from osmosis_ai.rollout.eval.common import ExternalLLMClient

# OpenAI (simple format)
client = ExternalLLMClient("gpt-4o")

# Anthropic (LiteLLM format)
client = ExternalLLMClient("anthropic/claude-sonnet-4-20250514")

# With explicit API key
client = ExternalLLMClient(
    model="gpt-4o",
    api_key="sk-...",
)

# Local Ollama
client = ExternalLLMClient(
    model="ollama/llama2",
    api_base="http://localhost:11434",
)

# Use as async context manager
async with client:
    result = await client.chat_completions(messages, tools=tools)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model name (simple or LiteLLM format) |
| `api_key` | `Optional[str]` | `None` | API key (or use env var) |
| `api_base` | `Optional[str]` | `None` | Base URL for OpenAI-compatible APIs |

**Methods:**

#### `async chat_completions(messages, tools, **kwargs) -> CompletionsResult`

Call the LLM with messages and tools.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[Dict]` | Conversation history |
| `tools` | `List[Dict]` | Tool definitions (OpenAI format) |
| `**kwargs` | - | Additional params (temperature, max_tokens, etc.) |

---

### LocalTestRunner

Execute batch tests.

```python
from osmosis_ai.rollout.eval.test_mode import LocalTestRunner

runner = LocalTestRunner(
    agent_loop=MyAgentLoop(),
    llm_client=client,
    debug=True,
)

# Single row test
result = await runner.run_single(row, row_index=0)

# Batch test
batch_result = await runner.run_batch(
    rows=rows,
    max_turns=10,
    completion_params={"temperature": 0.7},
    on_progress=lambda current, total, result: print(f"[{current}/{total}]"),
    start_index=0,
)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Agent to test |
| `llm_client` | `ExternalLLMClient` | required | LLM client |
| `debug` | `bool` | `False` | Enable debug output |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logs |

**Methods:**

#### `async run_single(row, row_index, max_turns, completion_params) -> LocalTestRunResult`

Test a single row.

#### `async run_batch(rows, max_turns, completion_params, on_progress, start_index) -> LocalTestBatchResult`

Test multiple rows.

---

### LocalTestRunResult

Result from a single test run.

```python
@dataclass
class LocalTestRunResult:
    row_index: int                    # Dataset row index
    success: bool                     # Whether completed without error
    result: Optional[RolloutResult]   # Agent result (if successful)
    error: Optional[str]              # Error message (if failed)
    duration_ms: float                # Execution time
    token_usage: Dict[str, int]       # Token statistics
```

**Token Usage Fields:**

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `total_tokens` | Total tokens |
| `num_llm_calls` | Number of LLM calls |

---

### LocalTestBatchResult

Aggregated results from batch testing.

```python
@dataclass
class LocalTestBatchResult:
    results: List[LocalTestRunResult]  # Individual results
    total: int                         # Total rows tested
    passed: int                        # Successful completions
    failed: int                        # Failed tests
    total_duration_ms: float           # Total execution time
    total_tokens: int                  # Total tokens used
```

---

### InteractiveRunner

Step-by-step debugging runner.

```python
from osmosis_ai.rollout.eval.test_mode import InteractiveRunner

runner = InteractiveRunner(
    agent_loop=MyAgentLoop(),
    llm_client=client,
    debug=True,
)

await runner.run_interactive_session(
    rows=rows,
    max_turns=10,
    completion_params={"temperature": 0.7},
    initial_row=5,
    row_offset=0,
)
```

---

## Exceptions

All local workflow exceptions are shared by local commands (`test` / `bench`).

```python
from osmosis_ai.rollout.eval.common import (
    DatasetValidationError,
    DatasetParseError,
    ProviderError,
    ToolValidationError,
)
```

| Exception | Description |
|-----------|-------------|
| `DatasetValidationError` | Dataset missing required columns or invalid values |
| `DatasetParseError` | Failed to parse dataset file |
| `ProviderError` | LLM provider error (auth, rate limit, etc.) |
| `ToolValidationError` | Invalid tool schema |

### Example Error Handling

```python
from osmosis_ai.rollout.eval.common import (
    DatasetReader,
    DatasetValidationError,
    DatasetParseError,
)

try:
    reader = DatasetReader("./data.jsonl")
    rows = reader.read()
except DatasetParseError as e:
    print(f"Failed to parse file: {e}")
except DatasetValidationError as e:
    print(f"Invalid dataset: {e}")
```

---

## Output Format

When using `--output`, results are saved as JSON:

```json
{
  "summary": {
    "total": 100,
    "passed": 95,
    "failed": 5,
    "total_duration_ms": 45230,
    "total_tokens": 125000
  },
  "results": [
    {
      "row_index": 0,
      "success": true,
      "error": null,
      "duration_ms": 450,
      "token_usage": {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200,
        "num_llm_calls": 2
      },
      "reward": 1.0,
      "finish_reason": "stop"
    }
  ]
}
```

---

## Environment Variables

API keys can be set via environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Azure | `AZURE_API_KEY` |

See [LiteLLM Environment Variables](https://docs.litellm.ai/docs/providers) for more providers.

---

## Best Practices

1. **Start with a small dataset**: Use `--limit 5` to validate your setup before running full tests.

2. **Use interactive mode for debugging**: When a test fails, use `--interactive --row N` to step through execution.

3. **Save results for analysis**: Use `-o results.json` to save detailed metrics for later analysis.

4. **Set appropriate timeouts**: Complex agent loops may need longer `--max-turns` values.

5. **Monitor token usage**: Track `total_tokens` to estimate costs before running large batches.

---

## See Also

- [Bench Mode](./bench.md) - Benchmark agents with eval functions and pass@k
- [Architecture](./architecture.md) - System design overview
- [API Reference](./api-reference.md) - Complete SDK API documentation
- [Examples](./examples.md) - Working code examples
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) - Supported LLM providers