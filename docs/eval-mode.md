# Eval Mode

Evaluate your trained models by running agent implementations against datasets with custom eval functions, statistical analysis, and pass@k metrics. Works with both **Local Rollout** (MCP tools) and **Remote Rollout** (RolloutAgentLoop) agents.

## Quick Start

```bash
# Benchmark a trained model served at an endpoint
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1

# Local Rollout: evaluate MCP tools directory
osmosis eval --mcp ./mcp -d data.jsonl \
    --eval-fn reward_fn:compute_reward --model openai/gpt-5-mini

# Compare trained model vs baseline
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward \
    --model my-finetuned-model --base-url http://localhost:8000/v1 \
    --baseline-model openai/gpt-5-mini

# pass@k with concurrent execution
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 5 --batch-size 5 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

The `--mcp` directory must contain a `main.py` with a `FastMCP` instance. See [Local Rollout MCP Tools](./local-rollout/mcp-tools.md). `--mcp` and `-m` are mutually exclusive.

---

## Connecting to Model Endpoints

Eval mode works with any OpenAI-compatible serving endpoint via `--base-url`:

| Serving Platform | Example `--base-url` |
|-----------------|----------------------|
| osmosis-serving | `https://inference.osmosis.ai/v1` |
| vLLM | `http://localhost:8000/v1` |
| SGLang | `http://localhost:30000/v1` |
| Ollama | `http://localhost:11434/v1` |

The `--model` parameter should match the model name as registered in the serving endpoint. You can also use LiteLLM providers (e.g., `openai/gpt-5-mini`) as either the primary or baseline model.

---

## Eval Functions

Eval functions score agent outputs. Two signatures are supported, auto-detected by the first parameter name.

### Simple Mode (compatible with `@osmosis_reward`)

```python
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

- First parameter must be named `solution_str`
- Compatible with functions decorated with `@osmosis_reward`

### Full Mode

```python
def conversation_quality(messages: list, ground_truth: str, metadata: dict, **kwargs) -> float:
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    return min(1.0, len(assistant_messages) / 3)
```

- First parameter must be named `messages`
- Both sync and async functions are supported

---

## pass@k

When `--n` is greater than 1, eval mode runs each dataset row multiple times and computes pass@k metrics.

Formula: `pass@k = 1 - C(n-c, k) / C(n, k)` where `n` = total runs, `c` = passing runs (score >= `--pass-threshold`), `k` = sample size.

---

## Result Caching & Resume

Eval mode automatically caches results to disk so evaluations can be **interrupted and resumed** without losing progress. This is especially useful for long-running evaluations with many rows or multiple runs per row.

### How It Works

1. When an evaluation starts, a **task ID** is computed from the full configuration (model, dataset, eval functions, parameters, and source code fingerprints).
2. Results are written to a JSON cache file under `~/.cache/osmosis/eval/` (or `$OSMOSIS_CACHE_DIR/eval/`), organized by `{model}/{dataset}/`.
3. If the evaluation is interrupted (Ctrl+C, SIGTERM, or crash), re-running the **same command** automatically resumes from where it left off.
4. A file lock prevents concurrent evaluations with the same configuration from conflicting.

### Cache Invalidation

The cache is keyed on a fingerprint that includes:

- **Module source code** — changes to your agent's `.py` file (or entire package directory for packages, or MCP directory) invalidate the cache.
- **Eval function source code** — changes to any eval function's source file invalidate the cache.
- **Dataset content** — the dataset file is fingerprinted; any modification is detected.
- **All CLI parameters** — model, base URL, `--n`, `--max-turns`, `--pass-threshold`, `--temperature`, `--max-tokens`, `--offset`, `--limit`.

If any of these change, a new cache entry is created automatically.

> **Note**: Module fingerprinting covers the agent's own source file (or package directory). Changes to external dependencies (e.g., a library your agent imports) are **not** detected. Use `--fresh` to force a clean restart when external imports change.

### Resuming After Interruption

Simply re-run the exact same command:

```bash
# First run — interrupted at row 50/100
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score --model my-model
# ^C (interrupted)

# Second run — resumes from row 50
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score --model my-model
```

When resuming, the CLI displays:

```
Resuming eval (50/100 runs completed)
Note: Module fingerprint covers file server.py. External dependency changes
      are not detected. Use --fresh if you changed external imports.
```

### Dataset Integrity

During evaluation, the dataset file is periodically re-checked (every 100 runs or 5 minutes) to detect modifications. If the dataset changes mid-evaluation, the run stops with an error:

```
Error: Dataset was modified during evaluation. Results may be inconsistent. Use --fresh to restart.
```

If a completed evaluation is loaded from cache but the dataset has since changed, a warning is displayed alongside the cached results.

### Cache Management

```bash
# Print the cache root directory path
osmosis eval cache dir

# List all cached evaluations
osmosis eval cache ls

# List with filters
osmosis eval cache ls --model gpt-4
osmosis eval cache ls --status in_progress
osmosis eval cache ls --dataset my_data

# Remove a specific cached evaluation by task ID
osmosis eval cache rm <task_id>

# Remove all cached evaluations (with confirmation prompt)
osmosis eval cache rm --all

# Remove with filters (skip confirmation with -y)
osmosis eval cache rm --status in_progress --yes
osmosis eval cache rm --model gpt-4 --yes

# Force a fresh evaluation, discarding cached results
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score --model my-model --fresh

# Re-run only failed runs from a previous evaluation
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score --model my-model --retry-failed
```

The `--fresh` flag backs up existing cache files (with a `.backup.{timestamp}` suffix) before creating a new cache.

`--fresh` and `--retry-failed` are mutually exclusive.

---

## Conversation Logging

Use `--log-samples` to save the full conversation messages for each run to a JSONL file alongside the cache:

```bash
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score \
    --model my-model --log-samples
```

Each line in the samples file is a JSON object containing `row_index`, `run_index`, `model_tag` (if using baseline comparison), and the full `messages` array.

> **Note**: When resuming a previously interrupted evaluation, only new runs are logged. Prior runs from the cache do not have their messages retroactively saved. Use `--fresh --log-samples` if you need complete logs for all runs.

---

## Structured Output

Use `--output-path` to write results to a structured directory:

```bash
osmosis eval -m server:agent_loop -d data.jsonl --eval-fn rewards:score \
    --model my-model --output-path ./results
```

This creates:

```
results/
  {model}/
    {dataset}/
      results_{timestamp}_{task_id}.json
      samples_{timestamp}_{task_id}.jsonl   # if --log-samples is used
```

The results JSON uses the same schema as the internal cache file, with `status` always set to `"completed"`.

### Results JSON (`--output-path` / `-o`)

```json
{
  "version": 1,
  "task_id": "a3f1c9e8",
  "config_hash": "b7d2e4f1",
  "status": "completed",
  "created_at": "2026-03-03T14:22:01Z",
  "updated_at": "2026-03-03T14:25:13Z",
  "config": {
    "model": "my-finetuned-model",
    "base_url": "http://localhost:8000/v1",
    "n_runs": 8,
    "pass_threshold": 1.0,
    "eval_fns": ["compute_reward"],
    "max_turns": 10,
    "batch_size": 4
  },
  "runs": [
    {
      "row_index": 0,
      "run_index": 0,
      "success": true,
      "scores": {"compute_reward": 1.000},
      "duration_ms": 2345.6,
      "tokens": 512,
      "model_tag": null,
      "error": null
    },
    {
      "row_index": 0,
      "run_index": 1,
      "success": true,
      "scores": {"compute_reward": 0.750},
      "duration_ms": 1987.3,
      "tokens": 483,
      "model_tag": null,
      "error": null
    }
  ],
  "summary": {
    "total_runs": 160,
    "total_tokens": 89240,
    "total_duration_ms": 192400.0,
    "eval_fns": {
      "compute_reward": {
        "mean": 0.85,
        "std": 0.127,
        "min": 0.4,
        "max": 1.0,
        "pass_at_1": 0.85,
        "pass_at_2": 0.95,
        "pass_at_4": 1.0,
        "pass_at_8": 1.0
      }
    }
  }
}
```

### Samples JSONL (`--log-samples`)

Each line is a JSON object:

```jsonl
{"row_index": 0, "run_index": 0, "model_tag": null, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Solve: 2+2"}, {"role": "assistant", "content": "4"}]}
{"row_index": 0, "run_index": 1, "model_tag": null, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Solve: 2+2"}, {"role": "assistant", "content": "The answer is 4."}]}
{"row_index": 1, "run_index": 0, "model_tag": "primary", "messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]}
```

---

## Example Output

### pass@k (`--n`)

```bash
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --n 8 --batch-size 4 \
    --model my-finetuned-model --base-url http://localhost:8000/v1
```

```
osmosis-eval v0.2.17
Endpoint: http://localhost:8000/v1
Model: my-finetuned-model

Loading eval functions...
  compute_reward (simple mode: solution_str)

Running evaluation (20 rows x8 runs, batch_size=4)...

[1/160] OK (2.3s, 512 tokens) [compute_reward=1.000]
[2/160] OK (1.9s, 483 tokens) [compute_reward=0.750]
[3/160] OK (3.1s, 621 tokens) [compute_reward=1.000]
[4/160] FAILED (1.5s, 340 tokens) - Agent exceeded max turns
[5/160] OK (2.7s, 558 tokens) [compute_reward=1.000]
[6/160] OK (2.1s, 497 tokens) [compute_reward=0.500]
...

Evaluation Results:
  Total runs: 160
  Duration: 3m12.4s
  Total tokens: 89,240

  compute_reward: mean=0.850 median=0.900 std=0.127
    pass@1: 85.0%
    pass@2: 95.0%
    pass@4: 100.0%
    pass@8: 100.0%

Cache: ~/.cache/osmosis/eval/my-finetuned-model/data/20260303_142201_a3f1c9e8.json
Tip: Re-run the same command to resume if interrupted.
```

### Baseline Comparison (`--baseline-model`)

```bash
osmosis eval -m server:agent_loop -d data.jsonl \
    --eval-fn rewards:compute_reward --eval-fn rewards:format_score \
    --model my-finetuned-model --base-url http://localhost:8000/v1 \
    --baseline-model openai/gpt-5-mini
```

```
osmosis-eval v0.2.17
Endpoint: http://localhost:8000/v1
Model: my-finetuned-model
Baseline model: openai/gpt-5-mini

Loading eval functions...
  compute_reward (simple mode: solution_str)
  format_score (full mode: messages)

Running evaluation (20 rows, batch_size=1)...

[1/40] [primary] OK (2.3s, 512 tokens) [compute_reward=1.000, format_score=0.800]
[2/40] [baseline] OK (1.8s, 445 tokens) [compute_reward=0.750, format_score=0.900]
[3/40] [primary] OK (3.1s, 621 tokens) [compute_reward=1.000, format_score=0.850]
[4/40] [baseline] OK (2.0s, 498 tokens) [compute_reward=0.500, format_score=0.700]
...

Evaluation Results:
  Total runs: 40
  Duration: 1m45.2s
  Total tokens: 21,340

  [primary] my-finetuned-model:
  compute_reward: mean=0.900 median=0.950 std=0.105
  format_score: mean=0.830 median=0.850 std=0.090

  [baseline] openai/gpt-5-mini:
  compute_reward: mean=0.720 median=0.780 std=0.185
  format_score: mean=0.810 median=0.830 std=0.110
```

### Resuming After Interruption

First run — interrupted:

```
osmosis-eval v0.2.17
Endpoint: http://localhost:8000/v1
Model: my-finetuned-model

Running evaluation (20 rows x8 runs, batch_size=4)...

[1/160] OK (2.3s, 512 tokens) [compute_reward=1.000]
[2/160] OK (1.9s, 483 tokens) [compute_reward=0.750]
...
[45/160] OK (2.5s, 534 tokens) [compute_reward=1.000]
^C
Evaluation interrupted. Progress: 45/160 runs.
```

Second run — same command resumes automatically:

```
osmosis-eval v0.2.17
Endpoint: http://localhost:8000/v1
Model: my-finetuned-model

Resuming eval (45/160 runs completed)
Note: Module fingerprint covers file server.py. External dependency changes
      are not detected. Use --fresh if you changed external imports.

[46/160] OK (2.1s, 497 tokens) [compute_reward=0.850]
[47/160] OK (3.0s, 612 tokens) [compute_reward=1.000]
...

Evaluation Results:
  Total runs: 160
  Duration: 3m12.4s
  Total tokens: 89,240

  compute_reward: mean=0.850 median=0.900 std=0.127
    pass@1: 85.0%
    pass@2: 95.0%
    pass@4: 100.0%
    pass@8: 100.0%
```

---

## CLI Reference

```
osmosis eval [OPTIONS]
osmosis eval cache dir
```

### Required Options

| Option | Description |
|--------|-------------|
| `-d, --dataset FILE` | Path to dataset file (.parquet recommended, .jsonl, .csv) |
| `--eval-fn MODULE:FN` | Eval function path (can be specified multiple times) |
| `--model MODEL` | Model to benchmark |

### Agent Options (one required, mutually exclusive)

| Option | Description |
|--------|-------------|
| `-m, --module MODULE` | Module path to agent loop (format: `module:attribute`) |
| `--mcp DIR` | Path to MCP tools directory. Requires `pip install osmosis-ai[mcp]`. |

### Model Options

| Option | Description |
|--------|-------------|
| `--model MODEL` | Model name at the serving endpoint, or LiteLLM format (e.g., `openai/gpt-5-mini`) |
| `--base-url URL` | Base URL for the model serving endpoint |
| `--api-key KEY` | API key for the endpoint or LLM provider |

### Baseline Options (optional)

| Option | Description |
|--------|-------------|
| `--baseline-model MODEL` | Baseline model for comparison |
| `--baseline-base-url URL` | Base URL for the baseline model's endpoint |
| `--baseline-api-key KEY` | API key for the baseline model provider |

### Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n N` | `1` | Number of runs per row for pass@k |
| `--pass-threshold FLOAT` | `1.0` | Score >= threshold counts as pass |
| `--max-turns N` | `10` | Maximum agent turns per run |
| `--temperature FLOAT` | - | LLM sampling temperature |
| `--max-tokens N` | - | Maximum tokens per completion |
| `--batch-size N` | `1` | Number of concurrent runs |
| `--limit N` | all | Maximum rows to benchmark |
| `--offset N` | `0` | Number of rows to skip |

### Cache & Resume Options

| Option | Description |
|--------|-------------|
| `--fresh` | Force restart evaluation from scratch, discarding any cached results (backs up existing cache) |
| `--retry-failed` | Re-execute only failed runs from a previous evaluation. Mutually exclusive with `--fresh`. |

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output-path DIR` | Write results to structured directory (`{model}/{dataset}/results_{ts}_{id}.json`) |
| `--log-samples` | Save full conversation messages to a JSONL file alongside the cache |
| `-q, --quiet` | Suppress progress output |
| `--debug` | Enable debug logging |

### Cache Management Subcommands

| Command | Description |
|---------|-------------|
| `osmosis eval cache dir` | Print the cache root directory path |

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `EvalFnError` | Eval function loading, signature detection, or execution error |
| `TimeoutError` | Another evaluation with the same config is already running (lock contention) |
| `RuntimeError` | Cache version mismatch, config hash collision, or dataset fingerprint mismatch |

All other exceptions are shared with [Test Mode](./test-mode.md#exceptions).

---

## See Also

- [Test Mode](./test-mode.md) -- Test agent logic with external LLMs
- [Dataset Format](./datasets.md) -- Supported formats and required columns
- [Configuration](./configuration.md) -- Environment variables including cache settings
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) -- Supported LLM providers
