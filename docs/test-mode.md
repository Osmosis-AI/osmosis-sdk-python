# Test Mode

Test your agent implementations locally without TrainGate using external LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers). Works with both **Local Rollout** (MCP tools) and **Remote Rollout** (RolloutAgentLoop) agents.

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

Interactive mode allows step-by-step debugging of agent execution.

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

## Exceptions

All local workflow exceptions are shared by `test` and `eval` commands.

| Exception | Description |
|-----------|-------------|
| `DatasetValidationError` | Dataset missing required columns or invalid values |
| `DatasetParseError` | Failed to parse dataset file |
| `ProviderError` | LLM provider error (auth, rate limit, etc.) |
| `ToolValidationError` | Invalid tool schema |

---

## See Also

- [Eval Mode](./eval-mode.md) -- Evaluate agents with eval functions and pass@k
- [Dataset Format](./datasets.md) -- Supported formats and required columns
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) -- Supported LLM providers
