# Bench Mode

Benchmark your trained models by running `RolloutAgentLoop` implementations against datasets with custom eval functions, statistical analysis, and pass@k metrics.

## Overview

Bench mode is designed for **evaluating trained models**. Connect to any OpenAI-compatible model serving endpoint (such as osmosis-serving, vLLM, or SGLang) and measure agent performance with custom eval functions.

Key capabilities:

- **Benchmark trained models** against eval functions by connecting to serving endpoints
- Run multiple trials per row for pass@k analysis
- Use existing `@osmosis_reward` functions or full-context eval functions
- Get statistical summaries (mean, std, min, max) per eval function
- Compare model quality across checkpoints or configurations
- **Concurrent execution** with `--batch_size` for faster benchmarks
- Optionally use LiteLLM providers (e.g., `openai/gpt-4o`) as comparison baselines

## Quick Start

### Benchmarking a Trained Model

The primary use case is connecting to a model serving endpoint:

```bash
# Benchmark a trained model served at an endpoint
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model \
    --base-url http://localhost:8000/v1

# Benchmark against osmosis-serving
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-trained-model \
    --base-url https://inference.osmosis.ai/v1 \
    --api-key $OSMOSIS_API_KEY

# Multiple eval functions
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:exact_match \
    --eval-fn rewards:partial_match \
    --model my-finetuned-model \
    --base-url http://localhost:8000/v1
```

### Comparison Baselines with LiteLLM

You can also benchmark against external LLM providers using [LiteLLM](https://docs.litellm.ai/docs/providers) format for comparison:

```bash
# Compare against GPT-4o as a baseline
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward --model openai/gpt-4o

# Compare against Claude
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward --model anthropic/claude-sonnet-4-20250514
```

### Concurrent Execution

Use `--batch_size` to run multiple requests in parallel:

```bash
# Run 5 concurrent requests
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1 \
    --batch_size 5

# Combine with pass@k — 10 runs per row, 5 concurrent
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 10 --batch_size 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

### pass@k Analysis

```bash
# pass@k with 5 runs per row
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Custom pass threshold (default is 1.0)
osmosis bench -m my_agent:MyAgentLoop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 --pass-threshold 0.5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

### Programmatic Usage

```python
from osmosis_ai.rollout.eval.common import DatasetReader, ExternalLLMClient
from osmosis_ai.rollout.eval.bench import BenchRunner, EvalFnWrapper, load_eval_fns

# Load dataset
reader = DatasetReader("./test_data.jsonl")
rows = reader.read(limit=10)

# Connect to trained model endpoint
client = ExternalLLMClient(
    "my-finetuned-model",
    api_base="http://localhost:8000/v1",
)

# Load eval functions
eval_fns = load_eval_fns(["rewards:exact_match", "rewards:partial_match"])

# Create runner
runner = BenchRunner(
    agent_loop=MyAgentLoop(),
    llm_client=client,
    eval_fns=eval_fns,
)

# Run benchmark (batch_size=5 for concurrent execution)
async with client:
    result = await runner.run_bench(
        rows=rows,
        n_runs=5,
        max_turns=10,
        pass_threshold=0.5,
        completion_params={"temperature": 0.7},
        batch_size=5,
    )

# Print results
for name, summary in result.eval_summaries.items():
    print(f"{name}: mean={summary.mean:.3f}, std={summary.std:.3f}")
    for k, v in summary.pass_at_k.items():
        print(f"  pass@{k}: {v*100:.1f}%")
```

---

## Connecting to Model Endpoints

Bench mode works with any OpenAI-compatible serving endpoint via `--base-url`:

| Serving Platform | Example `--base-url` |
|-----------------|----------------------|
| osmosis-serving | `https://inference.osmosis.ai/v1` |
| vLLM | `http://localhost:8000/v1` |
| SGLang | `http://localhost:30000/v1` |
| Ollama | `http://localhost:11434/v1` |
| Any OpenAI-compatible API | `http://<host>:<port>/v1` |

The `--model` parameter should match the model name as registered in the serving endpoint. For example, if you deployed `my-org/my-finetuned-llama` with vLLM, use `--model my-org/my-finetuned-llama`.

---

## Dataset Format

Bench mode uses the same dataset format as [Test Mode](./test-mode.md#dataset-format). Each row must contain these columns (case-insensitive):

| Column | Type | Description |
|--------|------|-------------|
| `ground_truth` | `str` | Expected output (passed to eval functions) |
| `user_prompt` | `str` | User message to start the conversation |
| `system_prompt` | `str` | System prompt for the LLM |

Additional columns are passed to eval functions via `metadata` (full mode) or `extra_info` (simple mode).

---

## Eval Functions

Eval functions score agent outputs. Two signatures are supported, auto-detected by the first parameter name.

### Simple Mode (compatible with `@osmosis_reward`)

Use when you only need the final assistant response:

```python
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    """Score based on the last assistant message content."""
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

- First parameter must be named `solution_str`
- `solution_str` is extracted from the last assistant message
- `extra_info` contains the full row dict (all columns)
- Compatible with functions decorated with `@osmosis_reward`

### Full Mode

Use when you need the complete conversation history:

```python
def conversation_quality(messages: list, ground_truth: str, metadata: dict, **kwargs) -> float:
    """Score based on the full conversation."""
    # messages: full conversation history (system, user, assistant, tool messages)
    # metadata: full row dict (all columns)
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    return min(1.0, len(assistant_messages) / 3)
```

- First parameter must be named `messages`
- Receives the complete message list from the agent run
- Both sync and async functions are supported

### Async Eval Functions

```python
async def llm_judge(messages: list, ground_truth: str, metadata: dict, **kwargs) -> float:
    """Use an LLM to judge the conversation quality."""
    # async eval functions are awaited automatically
    score = await call_judge_llm(messages, ground_truth)
    return score
```

---

## pass@k

When `--n` is greater than 1, bench mode runs each dataset row multiple times and computes pass@k metrics.

**pass@k** estimates the probability that at least one of `k` randomly selected samples from `n` total runs passes (score >= threshold).

Formula: `pass@k = 1 - C(n-c, k) / C(n, k)`

Where:
- `n` = total runs per row
- `c` = number of passing runs (score >= `--pass-threshold`)
- `k` = sample size

pass@k is computed per row, then averaged across all rows.

### Example

With `--n 10 --pass-threshold 0.5`:
- If a row has 7/10 runs passing, pass@1 = 0.7, pass@3 = 0.97
- Final pass@k values are averages across all rows

---

## CLI Reference

```
osmosis bench [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `-m, --module MODULE` | Module path to agent loop (format: `module:attribute`) |
| `-d, --dataset FILE` | Path to dataset file (.json, .jsonl, .parquet) |
| `--eval-fn MODULE:FN` | Eval function path (can be specified multiple times) |
| `--model MODEL` | Model to benchmark (see [Model Options](#model-options)) |

### Model Options

| Option | Description |
|--------|-------------|
| `--model MODEL` | Required. Model name at the serving endpoint (e.g., `my-finetuned-model`), or LiteLLM provider format for baselines (e.g., `openai/gpt-4o`) |
| `--base-url URL` | Base URL for the model serving endpoint (e.g., `http://localhost:8000/v1`). Use this to connect to trained model endpoints. |
| `--api-key KEY` | API key for the endpoint or LLM provider (or use env var) |

### Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n N` | `1` | Number of runs per row for pass@k |
| `--pass-threshold FLOAT` | `1.0` | Score >= threshold counts as pass for pass@k |
| `--max-turns N` | `10` | Maximum agent turns per run |
| `--temperature FLOAT` | - | LLM sampling temperature |
| `--max-tokens N` | - | Maximum tokens per completion |
| `--batch_size N` | `1` | Number of concurrent runs |
| `--limit N` | all | Maximum rows to benchmark |
| `--offset N` | `0` | Number of rows to skip |

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Write results to JSON file |
| `-q, --quiet` | Suppress progress output |
| `--debug` | Enable debug logging |

### Examples

```bash
# Benchmark trained model at an endpoint
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Multiple eval functions
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:exact_match \
    --eval-fn rewards:semantic_similarity \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# pass@5 analysis
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Lenient pass threshold
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 --pass-threshold 0.5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Baseline comparison with external LLM, save results
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model openai/gpt-4o -o results.json

# Concurrent execution (5 runs at a time)
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --batch_size 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Benchmark subset
osmosis bench -m my_agent:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --limit 10 --offset 50 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

---

## API Reference

### BenchRunner

Orchestrates benchmark execution with eval function scoring.

```python
from osmosis_ai.rollout.eval.bench import BenchRunner

runner = BenchRunner(
    agent_loop=MyAgentLoop(),
    llm_client=client,
    eval_fns=eval_fns,
    debug=True,
)

# Single run
result = await runner.run_single(row, row_index=0, run_index=0)

# Full benchmark (batch_size for concurrent execution)
bench_result = await runner.run_bench(
    rows=rows,
    n_runs=5,
    max_turns=10,
    pass_threshold=0.5,
    completion_params={"temperature": 0.7},
    on_progress=lambda current, total, result: print(f"[{current}/{total}]"),
    batch_size=5,
)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Agent to benchmark |
| `llm_client` | `ExternalLLMClient` | required | LLM client |
| `eval_fns` | `List[EvalFnWrapper]` | required | Eval functions to apply |
| `debug` | `bool` | `False` | Enable debug logging |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logs |
| `llm_client_factory` | `Optional[Callable]` | `None` | Factory to create LLM client instances for concurrent execution. If `None`, clones from the provided `llm_client`. |

**Methods:**

#### `async run_single(row, row_index, run_index, max_turns, completion_params) -> BenchRunResult`

Run the agent once on a row and apply all eval functions.

#### `async run_bench(rows, n_runs, max_turns, completion_params, pass_threshold, on_progress, start_index, batch_size) -> BenchResult`

Run the full benchmark across all rows. When `batch_size > 1`, runs execute concurrently using a pool of independent LLM client instances.

---

### BenchRunResult

Result from a single agent run with eval scores.

```python
@dataclass
class BenchRunResult:
    run_index: int              # Which run (0-indexed within a row)
    success: bool               # Whether agent completed successfully
    scores: Dict[str, float]    # Eval function name -> score
    duration_ms: float          # Execution time in milliseconds
    tokens: int                 # Total tokens used
    error: Optional[str]        # Error message (if failed)
```

---

### BenchRowResult

All runs for a single dataset row.

```python
@dataclass
class BenchRowResult:
    row_index: int                  # Dataset row index
    runs: List[BenchRunResult]      # All runs for this row
```

---

### BenchResult

Full benchmark result with aggregated statistics.

```python
@dataclass
class BenchResult:
    rows: List[BenchRowResult]                      # Per-row results
    eval_summaries: Dict[str, BenchEvalSummary]     # Per-eval statistics
    total_rows: int                                  # Number of dataset rows
    total_runs: int                                  # Total runs (rows * n_runs)
    total_tokens: int                                # Total tokens consumed
    total_duration_ms: float                         # Total wall time
    n_runs: int                                      # Runs per row
    pass_threshold: float                            # Score threshold for pass@k
```

---

### BenchEvalSummary

Summary statistics for a single eval function.

```python
@dataclass
class BenchEvalSummary:
    mean: float                     # Mean score across all runs
    std: float                      # Standard deviation
    min: float                      # Minimum score
    max: float                      # Maximum score
    pass_at_k: Dict[int, float]     # k -> pass@k value (only when n > 1)
```

---

### EvalFnWrapper

Normalizes eval function signatures into a unified async interface.

```python
from osmosis_ai.rollout.eval.bench import EvalFnWrapper

wrapper = EvalFnWrapper(fn=my_eval_function, name="my_eval")

# Call with normalized arguments
score = await wrapper(messages, ground_truth, metadata)
```

---

### load_eval_fns

Load and wrap eval functions from module paths.

```python
from osmosis_ai.rollout.eval.bench import load_eval_fns

eval_fns = load_eval_fns(["rewards:exact_match", "rewards:partial_match"])
```

Each path must be in `module:function` format. Raises `EvalFnError` if loading fails or signatures are invalid.

---

## Output Format

When using `--output`, results are saved as JSON:

```json
{
  "config": {
    "model": "my-finetuned-model",
    "n_runs": 5,
    "pass_threshold": 0.5,
    "eval_fns": ["rewards:exact_match", "rewards:partial_match"]
  },
  "summary": {
    "total_rows": 100,
    "total_runs": 500,
    "eval_fns": {
      "rewards:exact_match": {
        "mean": 0.72,
        "std": 0.45,
        "min": 0.0,
        "max": 1.0,
        "pass_at_1": 0.72,
        "pass_at_3": 0.94,
        "pass_at_5": 0.98
      },
      "rewards:partial_match": {
        "mean": 0.85,
        "std": 0.22,
        "min": 0.3,
        "max": 1.0,
        "pass_at_1": 0.85,
        "pass_at_3": 0.99,
        "pass_at_5": 1.0
      }
    },
    "total_tokens": 625000,
    "total_duration_ms": 230500
  },
  "rows": [
    {
      "row_index": 0,
      "runs": [
        {
          "run_index": 0,
          "success": true,
          "scores": {
            "rewards:exact_match": 1.0,
            "rewards:partial_match": 1.0
          },
          "duration_ms": 450,
          "tokens": 200
        }
      ]
    }
  ]
}
```

---

## Exceptions

All exceptions are shared with [Test Mode](./test-mode.md#exceptions).

```python
from osmosis_ai.rollout.eval.common import (
    DatasetValidationError,
    DatasetParseError,
    ProviderError,
    ToolValidationError,
)
from osmosis_ai.rollout.eval.bench import EvalFnError
```

| Exception | Description |
|-----------|-------------|
| `EvalFnError` | Eval function loading, signature detection, or execution error |
| `DatasetValidationError` | Dataset missing required columns or invalid values |
| `DatasetParseError` | Failed to parse dataset file |
| `ProviderError` | LLM provider error (auth, rate limit, etc.) |
| `ToolValidationError` | Invalid tool schema |

---

## Environment Variables

See [Test Mode - Environment Variables](./test-mode.md#environment-variables) for API key configuration.

---

## Best Practices

1. **Start with `--n 1`**: Validate your eval functions work before running expensive pass@k analyses.

2. **Use `--base-url` for trained models**: Always point to your serving endpoint rather than relying on external APIs when evaluating your own models.

3. **Use multiple eval functions**: Evaluate different quality dimensions (correctness, efficiency, format compliance).

4. **Choose appropriate thresholds**: `--pass-threshold 1.0` (default) requires perfect scores. Use a lower threshold like `0.5` for partial-credit eval functions.

5. **Compare against baselines**: Run the same benchmark with LiteLLM providers (e.g., `--model openai/gpt-4o`) to establish baseline performance.

6. **Reuse `@osmosis_reward` functions**: Existing reward functions work as eval functions in simple mode without modification.

7. **Use `--batch_size` to speed up benchmarks**: Concurrent execution can significantly reduce wall time, especially with high-latency endpoints. Start with a moderate value (e.g., `5`) and increase based on endpoint capacity.

8. **Save results**: Use `-o results.json` to persist detailed per-run data for further analysis.

---

## See Also

- [Test Mode](./test-mode.md) - Test agent logic with external LLMs
- [Architecture](./architecture.md) - System design overview
- [API Reference](./api-reference.md) - Complete SDK API documentation
- [Examples](./examples.md) - Working code examples
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) - Supported LLM providers for baseline comparisons
