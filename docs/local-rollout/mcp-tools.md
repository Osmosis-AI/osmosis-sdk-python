# MCP Tools

In Local Rollout mode, the agent's callable tools are defined as [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) tool functions using [FastMCP](https://github.com/jlowin/fastmcp). During training, Osmosis loads your tools and makes them available to the LLM in the agent loop.

## Defining Tools

Tools are Python functions decorated with `@mcp.tool()`. Type hints and docstrings are used to generate the tool schema automatically.

```python
# mcp/tools/math.py
from server import mcp

@mcp.tool()
def multiply(first_val: float, second_val: float) -> float:
    """Calculate the product of two numbers.

    Args:
        first_val: the first value to be multiplied
        second_val: the second value to be multiplied
    """
    return round(first_val * second_val, 4)
```

Keep docstrings clear and descriptive -- the LLM uses them to decide when and how to call your tools.

## Folder Structure

The `mcp/` directory must contain a `main.py` entry point that creates a `FastMCP` instance and imports all tool modules:

```
mcp/
├── main.py              # Entry point
├── server/
│   └── mcp_server.py   # Creates the FastMCP instance
└── tools/
    ├── __init__.py      # Imports all tool modules
    └── math.py          # @mcp.tool() functions
```

For small projects, you can put everything in a single `main.py`:

```python
from fastmcp import FastMCP

mcp = FastMCP("my_tools")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)
```

## Testing MCP Tools Locally

```bash
pip install "osmosis-ai[mcp]"

# Batch test
osmosis test --mcp ./mcp -d test_data.jsonl --model openai/gpt-5-mini

# Interactive debugging
osmosis test --mcp ./mcp -d test_data.jsonl --interactive

# Evaluate with a reward function
osmosis eval --mcp ./mcp -d test_data.jsonl \
    --eval-fn reward_fn.compute_reward:numbers_match_reward \
    --model openai/gpt-5-mini
```

### How `--mcp` Works

When you pass `--mcp ./mcp`, the SDK:

1. Imports `mcp/main.py`, which triggers all `@mcp.tool()` registrations
2. Discovers registered tools and converts them to OpenAI function-calling schemas
3. Runs a built-in agent loop that calls the LLM, executes tool calls against your MCP functions, and repeats until the LLM stops calling tools or `--max-turns` is reached

> **Note:** `--mcp` and `-m/--module` are mutually exclusive. Use `--mcp` for Local Rollout projects; use `-m` for Remote Rollout projects that implement `RolloutAgentLoop`.

## Example Repository

See the complete working example: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)

## See Also

- [Test Mode](../test-mode.md) -- full documentation for `osmosis test`
- [Eval Mode](../eval-mode.md) -- full documentation for `osmosis eval`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
