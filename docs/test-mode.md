# Test Mode

Test your agent implementations locally without TrainGate using external LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers). Works with both **Local Rollout** (MCP tools) and **Remote Rollout** (RolloutAgentLoop) agents.

> **Note:** Breaking change: the legacy import path `osmosis_ai.rollout.test_mode` was removed.
> Use `osmosis_ai.rollout.eval.common` and `osmosis_ai.rollout.eval.test_mode` instead.

## Overview

Test mode enables you to:

- Validate agent logic before deploying to training infrastructure
- **Two agent modes**: provide a `RolloutAgentLoop` with `-m`, or load MCP tools directly with `--mcp`
- Use 100+ LLM providers (OpenAI, Anthropic, Groq, Ollama, etc.)
- Run batch tests against datasets with detailed metrics
- Debug step-by-step with interactive mode

## Quick Start

```bash
# Remote Rollout: batch test with OpenAI
osmosis test -m server:agent_loop -d data.jsonl --model gpt-5-mini

# Local Rollout: test MCP tools
osmosis test --mcp ./mcp -d data.jsonl --model openai/gpt-5-mini

# Interactive debugging (start at specific row)
osmosis test -m server:agent_loop -d data.jsonl --interactive --row 5
```

The `--mcp` directory must contain a `main.py` with a `FastMCP` instance. See [Local Rollout MCP Tools](./local-rollout/mcp-tools.md). `--mcp` and `-m` are mutually exclusive.

### Programmatic Usage

```python
from osmosis_ai.rollout.eval.common import DatasetReader, ExternalLLMClient
from osmosis_ai.rollout.eval.test_mode import LocalTestRunner

reader = DatasetReader("./test_data.jsonl")
rows = reader.read(limit=10)
client = ExternalLLMClient("gpt-5-mini")

runner = LocalTestRunner(agent_loop=CalculatorAgentLoop(), llm_client=client)
async with client:
    results = await runner.run_batch(rows=rows, max_turns=10)

print(f"Passed: {results.passed}/{results.total}")
```

---

## Dataset Format

See [Dataset Format](./datasets.md) for supported formats and required columns.

---

## CLI Reference

```
osmosis test [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `-d, --dataset FILE` | Path to dataset file (.parquet recommended, .jsonl, .csv) |

### Agent Options (one required, mutually exclusive)

| Option | Description |
|--------|-------------|
| `-m, --module, --agent MODULE` | Module path to agent loop (format: `module:attribute`) |
| `--mcp DIR` | Path to MCP tools directory. Requires `pip install osmosis-ai[mcp]`. |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model MODEL` | `gpt-5-mini` | Model name (see [Model Format](#model-format)) |
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

- **Simple**: `gpt-5-mini` (auto-prefixed to `openai/gpt-5-mini`)
- **LiteLLM format**: `provider/model` (e.g., `anthropic/claude-sonnet-4-5`)

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for supported providers.

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

---

## API Reference

### DatasetReader

```python
from osmosis_ai.rollout.eval.common import DatasetReader

reader = DatasetReader("./data.jsonl")
rows = reader.read(limit=10, offset=20)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `read(limit, offset)` | `List[DatasetRow]` | Read rows with optional pagination |
| `iter_rows()` | `Iterator[DatasetRow]` | Memory-efficient iterator |
| `__len__()` | `int` | Total row count |

`DatasetRow` is a `TypedDict` with `ground_truth`, `user_prompt`, `system_prompt`, plus any extra columns from the dataset.

### ExternalLLMClient

```python
from osmosis_ai.rollout.eval.common import ExternalLLMClient

client = ExternalLLMClient("gpt-5-mini")                          # OpenAI
client = ExternalLLMClient("anthropic/claude-sonnet-4-5")          # Anthropic
client = ExternalLLMClient("ollama/llama3.1", api_base="http://localhost:11434")  # Local
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model name (simple or LiteLLM format) |
| `api_key` | `Optional[str]` | `None` | API key (or use env var) |
| `api_base` | `Optional[str]` | `None` | Base URL for OpenAI-compatible APIs |

Method: `async chat_completions(messages, tools, **kwargs) -> CompletionsResult`

### LocalTestRunner

```python
from osmosis_ai.rollout.eval.test_mode import LocalTestRunner

runner = LocalTestRunner(agent_loop=agent, llm_client=client)
result = await runner.run_single(row, row_index=0)
batch_result = await runner.run_batch(rows=rows, max_turns=10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Agent to test |
| `llm_client` | `ExternalLLMClient` | required | LLM client |
| `debug` | `bool` | `False` | Enable debug output |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logs |

### Result Types

| Type | Key Fields |
|------|------------|
| `LocalTestRunResult` | `row_index`, `success`, `result: Optional[RolloutResult]`, `error`, `duration_ms`, `token_usage` |
| `LocalTestBatchResult` | `results`, `total`, `passed`, `failed`, `total_duration_ms`, `total_tokens` |

`token_usage` dict contains: `prompt_tokens`, `completion_tokens`, `total_tokens`, `num_llm_calls`.

### InteractiveRunner

```python
from osmosis_ai.rollout.eval.test_mode import InteractiveRunner

runner = InteractiveRunner(agent_loop=agent, llm_client=client)
await runner.run_interactive_session(rows=rows, max_turns=10, initial_row=5)
```

---

## Exceptions

All local workflow exceptions are shared by `test` and `eval` commands.

| Exception | Description |
|-----------|-------------|
| `DatasetValidationError` | Dataset missing required columns or invalid values |
| `DatasetParseError` | Failed to parse dataset file |
| `ProviderError` | LLM provider error (auth, rate limit, etc.) |
| `ToolValidationError` | Invalid tool schema |

---

## Output Format

When using `--output`, results are saved as JSON with this structure:

- `summary` -- total, passed, failed, total_duration_ms, total_tokens
- `results[]` -- per-row: row_index, success, error, duration_ms, token_usage, reward, finish_reason

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

## See Also

- [Eval Mode](./eval-mode.md) -- Evaluate agents with eval functions and pass@k
- [Dataset Format](./datasets.md) -- Supported formats and required columns
- [Remote Rollout Examples](./remote-rollout/examples.md) -- Working code examples
- [Local Rollout MCP Tools](./local-rollout/mcp-tools.md) -- MCP tool definition
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) -- Supported LLM providers
