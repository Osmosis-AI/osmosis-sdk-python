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

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Write results to JSON file |
| `-q, --quiet` | Suppress progress output |
| `--debug` | Enable debug logging |

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `EvalFnError` | Eval function loading, signature detection, or execution error |

All other exceptions are shared with [Test Mode](./test-mode.md#exceptions).

---

## See Also

- [Test Mode](./test-mode.md) -- Test agent logic with external LLMs
- [Dataset Format](./datasets.md) -- Supported formats and required columns
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) -- Supported LLM providers
