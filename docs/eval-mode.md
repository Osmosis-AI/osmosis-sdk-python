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

The legacy `-o`/`--output` flag is still supported and writes a single JSON file with the original nested format.

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
| `-o, --output FILE` | Write results to JSON file (legacy format) |
| `--output-path DIR` | Write results to structured directory (`{model}/{dataset}/results_{ts}_{id}.json`) |
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
