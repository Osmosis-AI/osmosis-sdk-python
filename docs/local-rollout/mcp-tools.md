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

The tool name, description, and parameter schemas are derived from the function name, docstring, and type annotations. Keep docstrings clear and descriptive -- the LLM uses them to decide when and how to call your tools.

## Folder Structure

The `mcp/` directory must contain a `main.py` entry point that creates a `FastMCP` instance and imports all tool modules:

```
mcp/
├── main.py              # Entry point
├── server/
│   ├── __init__.py
│   └── mcp_server.py   # Creates the FastMCP instance
└── tools/
    ├── __init__.py      # Imports all tool modules
    ├── math.py          # @mcp.tool() functions
    ├── api_helpers.py
    └── ml_utils.py
```

### main.py

```python
# mcp/main.py
from server import mcp
from tools import *    # Triggers @mcp.tool() registrations

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    mcp.run(transport="http", host=args.host, port=args.port)
```

### server/mcp_server.py

```python
# mcp/server/mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("OsmosisTools")
```

### tools/__init__.py

```python
# mcp/tools/__init__.py
from .math import *
from .api_helpers import *
```

## Simpler Structure

For small projects, you can put everything in a single `main.py`:

```python
# mcp/main.py
from fastmcp import FastMCP

mcp = FastMCP("my_tools")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)

@mcp.tool()
def lookup(key: str) -> str:
    """Look up a value by key."""
    data = {"pi": "3.14159", "e": "2.71828"}
    return data.get(key, "not found")
```

## Testing MCP Tools Locally

Use `osmosis test` with the `--mcp` flag to test your tools against a dataset:

```bash
# Install MCP support
pip install "osmosis-ai[mcp]"

# Batch test
osmosis test --mcp ./mcp -d test_data.jsonl --model openai/gpt-5-mini

# Interactive debugging -- step through each LLM call
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

This means you can iterate on tools locally, then push to GitHub for Osmosis to sync -- no `RolloutAgentLoop` code needed.

> **Note:** `--mcp` and `-m/--module` are mutually exclusive. Use `--mcp` for Local Rollout projects; use `-m` for Remote Rollout projects that implement `RolloutAgentLoop`.

## Example Repository

See the complete working example with MCP tools, reward functions, and rubrics: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)

## See Also

- [Test Mode](../test-mode.md) -- full documentation for `osmosis test`
- [Eval Mode](../eval-mode.md) -- full documentation for `osmosis eval`
- [Local Rollout Overview](./overview.md) -- repository structure and setup
- [Dataset Format](../datasets.md) -- dataset format for testing and evaluation
