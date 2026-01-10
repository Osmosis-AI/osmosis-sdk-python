# Osmosis Remote Rollout SDK

A lightweight SDK for integrating agent frameworks with the Osmosis remote rollout protocol. This SDK handles protocol communication between your agent logic and the Osmosis training infrastructure.

## Overview

The Remote Rollout SDK enables you to:

- Define custom agent loops that integrate with Osmosis training
- Handle LLM completions through the TrainGate service
- Execute tools and collect training data (trajectories, token IDs, logprobs)
- Run agents as scalable HTTP servers

## Installation

```bash
# Basic installation
pip install osmosis-ai

# With server support (FastAPI + uvicorn)
pip install osmosis-ai[server]

# Optional: environment-driven settings (.env / env vars)
pip install osmosis-ai[config]

# Full installation with all optional features
pip install osmosis-ai[full]
```

## Quick Start

### Step 1: Implement Your Agent Loop

Create a class that inherits from `RolloutAgentLoop` and implements two methods:

```python
from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutResult,
    RolloutRequest,
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
)

class MyAgentLoop(RolloutAgentLoop):
    name = "my_agent"  # Required: unique identifier

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        """Return tools available for this rollout."""
        return [
            OpenAIFunctionToolSchema(
                type="function",
                function=OpenAIFunctionSchema(
                    name="search",
                    description="Search for information",
                ),
            )
        ]

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        """Execute the agent loop."""
        messages = list(ctx.request.messages)

        for _ in range(ctx.request.max_turns):
            # Call LLM through TrainGate
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

            # Check if done
            if not result.has_tool_calls:
                break

            # Execute tools and add results
            for tool_call in result.tool_calls:
                tool_result = await self.execute_tool(tool_call)
                messages.append(tool_result)
                ctx.record_tool_call()

        return ctx.complete(messages)
```

### Step 2: Run the Server

#### Option 1: Using CLI (Recommended)

Export an instance of your agent loop and use the CLI:

```python
# my_agent.py
agent_loop = MyAgentLoop()
```

```bash
# Validate agent loop (checks tools, async run method, etc.)
osmosis validate -m my_agent:agent_loop

# Start server with validation (default port 9000)
osmosis serve -m my_agent:agent_loop

# Specify port
osmosis serve -m my_agent:agent_loop -p 8080

# Skip validation (not recommended)
osmosis serve -m my_agent:agent_loop --no-validate

# Enable auto-reload for development
osmosis serve -m my_agent:agent_loop --reload
```

#### Option 2: Using create_app()

Create a FastAPI application manually:

```python
from osmosis_ai.rollout import create_app

app = create_app(MyAgentLoop())
```

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

#### Option 3: Using serve_agent_loop()

Start programmatically with validation:

```python
from osmosis_ai.rollout import serve_agent_loop

# Validates and starts server
serve_agent_loop(MyAgentLoop(), port=9000)

# Skip validation (not recommended)
serve_agent_loop(MyAgentLoop(), port=9000, validate=False)
```

Your server is now ready to receive rollout requests from TrainGate.

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

Key methods:

- `ctx.chat(messages, **kwargs)`: Call the LLM
- `ctx.complete(messages)`: Return successful result
- `ctx.error(message)`: Return error result
- `ctx.record_tool_call()`: Track tool execution metrics

### RolloutResult

The return value from your agent loop:

- `status`: "COMPLETED" or "ERROR"
- `final_messages`: The complete conversation history
- `finish_reason`: Why the rollout ended
- `metrics`: Execution metrics (latency, token counts)

### OpenAIFunctionToolSchema

OpenAI-compatible tool definition format. Define your tools using:

```python
OpenAIFunctionToolSchema(
    type="function",
    function=OpenAIFunctionSchema(
        name="tool_name",
        description="What this tool does",
        parameters=OpenAIFunctionParametersSchema(
            properties={
                "arg1": OpenAIFunctionPropertySchema(type="string"),
            },
            required=["arg1"],
        ),
    ),
)
```

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

When installed with `osmosis-ai[config]`, settings can be loaded from environment variables (or a `.env` file) using these prefixes:

- `OSMOSIS_ROLLOUT_CLIENT_*` - Client settings (timeout, retries, etc.)
- `OSMOSIS_ROLLOUT_SERVER_*` - Server settings (concurrency, TTL, etc.)

Example:

```bash
export OSMOSIS_ROLLOUT_CLIENT_TIMEOUT_SECONDS=120
export OSMOSIS_ROLLOUT_SERVER_MAX_CONCURRENT_ROLLOUTS=200
```

## Next Steps

- [Architecture](./architecture.md) - Understand the system design
- [API Reference](./api-reference.md) - Complete API documentation
- [Examples](./examples.md) - Full working examples
- [Test Mode](./test-mode.md) - Local testing with external LLM providers
