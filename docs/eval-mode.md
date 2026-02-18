# Eval Mode

Evaluate your trained models by running agent implementations against datasets with custom eval functions, statistical analysis, and pass@k metrics. Works with both **Local Rollout** (MCP tools) and **Remote Rollout** (RolloutAgentLoop) agents.

## Overview

Eval mode is designed for **evaluating trained models**. Connect to any OpenAI-compatible model serving endpoint (such as osmosis-serving, vLLM, or SGLang) and measure agent performance with custom eval functions.

Key capabilities:

- **Benchmark trained models** against eval functions by connecting to serving endpoints
- **Two agent modes**: provide a `RolloutAgentLoop` with `-m`, or load MCP tools directly with `--mcp`
- Run multiple trials per row for pass@k analysis
- Use existing `@osmosis_reward` functions or full-context eval functions
- Get statistical summaries (mean, std, min, max) per eval function
- Compare model quality with `--baseline-model`
- **Concurrent execution** with `--batch-size` for faster benchmarks

> **Note:** Command split (development-stage breaking change):
> `osmosis eval-rubric` is for hosted rubric evaluation, while `osmosis eval` is for agent eval mode documented here.

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

The `--model` parameter should match the model name as registered in the serving endpoint. You can also use LiteLLM providers (e.g., `openai/gpt-5-mini`, `anthropic/claude-sonnet-4-5`) as either the primary or baseline model.

---

## Dataset Format

See [Dataset Format](./datasets.md) for supported formats and required columns.

---

## Eval Functions

Eval functions score agent outputs. Two signatures are supported, auto-detected by the first parameter name.

### Simple Mode (compatible with `@osmosis_reward`)

Use when you only need the final assistant response:

```python
def exact_match(solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> float:
    return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
```

- First parameter must be named `solution_str`
- `extra_info` contains the full row dict (all columns)
- Compatible with functions decorated with `@osmosis_reward`

### Full Mode

Use when you need the complete conversation history:

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

Formula: `pass@k = 1 - C(n-c, k) / C(n, k)`

Where:
- `n` = total runs per row
- `c` = number of passing runs (score >= `--pass-threshold`)
- `k` = sample size

pass@k is computed per row, then averaged across all rows.

---

## CLI Reference

```
osmosis eval [OPTIONS]
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
| `--baseline-model MODEL` | Baseline model for comparison (reports per-model summary statistics) |
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

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Write results to JSON file |
| `-q, --quiet` | Suppress progress output |
| `--debug` | Enable debug logging |

---

## API Reference

### EvalRunner

Orchestrates benchmark execution with eval function scoring.

```python
from osmosis_ai.rollout.eval.evaluation import EvalRunner

runner = EvalRunner(
    agent_loop=agent,
    llm_client=client,
    eval_fns=eval_fns,
    baseline_llm_client=baseline_client,  # optional
)

result = await runner.run_eval(
    rows=rows, n_runs=5, max_turns=10,
    pass_threshold=0.5, batch_size=5,
    completion_params={"temperature": 0.7},
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
| `llm_client_factory` | `Optional[Callable]` | `None` | Factory for concurrent LLM client instances |
| `baseline_llm_client` | `Optional[ExternalLLMClient]` | `None` | Baseline LLM client for comparison mode |
| `baseline_llm_client_factory` | `Optional[Callable]` | `None` | Factory for concurrent baseline LLM client instances |

**Methods:**

- `async run_single(row, row_index, run_index, max_turns, completion_params) -> EvalRunResult`
- `async run_eval(rows, n_runs, max_turns, completion_params, pass_threshold, on_progress, start_index, batch_size) -> EvalResult`

### Result Types

| Type | Key Fields |
|------|------------|
| `EvalRunResult` | `run_index`, `success`, `scores: Dict[str, float]`, `duration_ms`, `tokens`, `error`, `model_tag` |
| `EvalRowResult` | `row_index`, `runs: List[EvalRunResult]` |
| `EvalResult` | `rows`, `eval_summaries`, `total_rows`, `total_runs`, `total_tokens`, `n_runs`, `pass_threshold`, `model_summaries` |
| `EvalEvalSummary` | `mean`, `std`, `min`, `max`, `pass_at_k: Dict[int, float]` |
| `EvalModelSummary` | `model`, `model_tag`, `eval_summaries`, `total_runs`, `total_tokens`, `total_duration_ms` |

### load_eval_fns

```python
from osmosis_ai.rollout.eval.evaluation import load_eval_fns

eval_fns = load_eval_fns(["rewards:exact_match", "rewards:partial_match"])
```

Each path must be in `module:function` format. Raises `EvalFnError` if loading fails.

---

## Output Format

When using `--output`, results are saved as JSON with this structure:

- `config` -- model, n_runs, pass_threshold, eval_fns, baseline_model
- `summary` -- total_rows, total_runs, per-eval stats (mean/std/min/max/pass_at_k), total_tokens
- `rows[]` -- per-row results with individual run scores, duration, tokens, and `model_tag`
- `model_summaries[]` -- per-model stats (only when `--baseline-model` is used)

---

## Exceptions

All exceptions are shared with [Test Mode](./test-mode.md#exceptions).

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

## See Also

- [Test Mode](./test-mode.md) -- Test agent logic with external LLMs
- [Dataset Format](./datasets.md) -- Supported formats and required columns
- [Remote Rollout Examples](./remote-rollout/examples.md) -- Working code examples
- [Local Rollout MCP Tools](./local-rollout/mcp-tools.md) -- MCP tool definition
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) -- Supported LLM providers
