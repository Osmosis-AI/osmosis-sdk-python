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

# Optional: observability extras
pip install osmosis-ai[logging]       # structured logging (structlog)
pip install osmosis-ai[tracing]       # OpenTelemetry tracing
pip install osmosis-ai[metrics]       # Prometheus metrics
pip install osmosis-ai[observability] # logging + tracing + metrics
pip install osmosis-ai[full]          # everything
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

### Step 2: Create the Server

Use `create_app()` to create a FastAPI application:

```python
from osmosis_ai.rollout import create_app

app = create_app(MyAgentLoop())
```

### Step 3: Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
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
| `/metrics` | GET | Prometheus metrics (when enabled) |

## Configuration Options

```python
app = create_app(
    agent_loop,
    max_concurrent=100,      # Max concurrent rollouts
    record_ttl_seconds=3600, # How long to keep completed records
    enable_metrics_endpoint=True,  # Expose /metrics when metrics are enabled
)
```

## Settings (Environment Variables)

When installed with `osmosis-ai[config]`, settings can be loaded from environment variables (or a `.env` file) using these prefixes:

- `OSMOSIS_ROLLOUT_CLIENT_*`
- `OSMOSIS_ROLLOUT_SERVER_*`
- `OSMOSIS_ROLLOUT_LOG_*`
- `OSMOSIS_ROLLOUT_TRACE_*`
- `OSMOSIS_ROLLOUT_METRICS_*`

Example:

```bash
export OSMOSIS_ROLLOUT_LOG_LEVEL=INFO
export OSMOSIS_ROLLOUT_METRICS_ENABLED=true
export OSMOSIS_ROLLOUT_METRICS_PREFIX=my_agent
```

## Next Steps

- [Architecture](./architecture.md) - Understand the system design
- [API Reference](./api-reference.md) - Complete API documentation
- [Examples](./examples.md) - Full working examples
