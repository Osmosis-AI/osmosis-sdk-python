# Osmosis Remote Rollout SDK

A lightweight SDK for integrating agent frameworks with the Osmosis remote rollout protocol. This SDK handles protocol communication between your agent logic and the Osmosis training infrastructure.

## Overview

The Remote Rollout SDK enables you to:

- Define custom agent loops that integrate with Osmosis training
- Handle LLM completions through the TrainGate service
- Execute tools and collect training data (trajectories, token IDs, logprobs)
- Run agents as scalable HTTP servers

For an alternative approach that requires no server infrastructure, see [Local Rollout](../local-rollout/overview.md).

## Installation

```bash
# Basic installation
pip install osmosis-ai

# With server support (FastAPI + uvicorn)
pip install osmosis-ai[server]

# Full installation with all optional features
pip install osmosis-ai[full]
```

## Quick Start

### Project Structure

```
rollout-server/
├── server.py        # Agent loop + FastAPI app
├── tools.py         # Tool definitions and execution
├── rewards.py       # Reward computation
├── test_data.jsonl  # Test dataset
└── pyproject.toml   # Dependencies: osmosis-ai[server]>=0.2.14
```

For the complete working project, see [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example).

### Step 1: Implement Your Agent Loop

Create `server.py` with a class that inherits from `RolloutAgentLoop`:

```python
class CalculatorAgentLoop(RolloutAgentLoop):
    name = "calculator"

    def get_tools(self, request: RolloutRequest):
        return CALCULATOR_TOOL_SCHEMAS

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)
        for _turn in range(ctx.request.max_turns):
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)
            if not result.has_tool_calls:
                break
            tool_results = await execute_tools(result.tool_calls)
            messages.extend(tool_results)
        return ctx.complete(messages, reward=compute_reward(...))

agent_loop = CalculatorAgentLoop()
app = create_app(agent_loop)
```

See [Examples](./examples.md) for the complete `tools.py` and `rewards.py` files.

### Step 2: Run the Server

```bash
# Validate agent loop (checks tools, async run method, etc.)
osmosis validate -m server:agent_loop

# Start server with validation (default port 9000)
osmosis serve -m server:agent_loop

# Specify port and enable auto-reload
osmosis serve -m server:agent_loop -p 8080 --reload
```

For programmatic alternatives (`create_app()`, `serve_agent_loop()`), see [Agent Loop Guide](./agent-loop.md#server).

## Key Concepts

### RolloutAgentLoop

The abstract base class for your agent implementation. You must:

- Set the `name` class attribute (unique identifier)
- Implement `get_tools()` to return available tools
- Implement `run()` to execute your agent logic

### RolloutContext

Provided to your `run()` method, containing:

- `request`: The original `RolloutRequest` with messages and params
- `tools`: List of tools returned by `get_tools()`
- `llm`: The `OsmosisLLMClient` for LLM calls

Key methods: `ctx.chat()`, `ctx.complete()`, `ctx.error()`, `ctx.record_tool_call()`

### RolloutResult

The return value from your agent loop:

- `status`: "COMPLETED" or "ERROR"
- `final_messages`: The complete conversation history
- `finish_reason`: Why the rollout ended
- `metrics`: Execution metrics (latency, token counts)

## Server Endpoints

The server created by `create_app()` exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/rollout/init` | POST | Accept rollout requests (returns 202) |
| `/health` | GET | Health check |

## Configuration Options

```python
app = create_app(
    agent_loop,
    max_concurrent=100,      # Max concurrent rollouts
    record_ttl_seconds=3600, # How long to keep completed records
)
```

## Settings (Environment Variables)

When installed with `osmosis-ai[server]`, settings can be loaded from environment variables (or a `.env` file) using these prefixes:

- `OSMOSIS_ROLLOUT_CLIENT_*` - Client settings (timeout, retries, etc.)
- `OSMOSIS_ROLLOUT_SERVER_*` - Server settings (concurrency, TTL, etc.)

See [Configuration](../configuration.md) for the full list of environment variables.

## Example Repository

See the complete working example: [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example)

## Next Steps

- [Architecture](./architecture.md) -- Understand the system design
- [Agent Loop Guide](./agent-loop.md) -- Complete API documentation
- [Examples](./examples.md) -- Full working examples
- [Testing](./testing.md) -- Unit tests and mock trainer
- [Dataset Format](../datasets.md) -- Supported formats and required columns
- [Test Mode](../test-mode.md) -- Local testing with external LLM providers
