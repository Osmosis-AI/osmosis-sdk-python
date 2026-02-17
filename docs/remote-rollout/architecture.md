# Architecture

This document describes the architecture and communication protocol of the Osmosis Remote Rollout SDK.

## System Overview

The Remote Rollout system consists of two main components:

```
┌─────────────┐                      ┌─────────────────┐
│  TrainGate  │ ◄──── HTTP ────────► │  RolloutServer  │
│  (Trainer)  │                      │  (Your Agent)   │
└─────────────┘                      └─────────────────┘
```

- **TrainGate**: The Osmosis training infrastructure that provides LLM generation and collects training data
- **RolloutServer**: Your server running the agent logic (using this SDK)

## Communication Protocol

### 1. Initialization Flow

```
TrainGate                           RolloutServer
    │                                     │
    │  POST /v1/rollout/init              │
    │  {rollout_id, server_url,           │
    │   messages, completion_params}      │
    ├────────────────────────────────────►│
    │                                     │
    │          202 Accepted               │
    │  {rollout_id, tools: [...]}         │
    │◄────────────────────────────────────┤
    │                                     │
```

TrainGate sends a `RolloutRequest` to `/v1/rollout/init`. RolloutServer:
1. Calls `agent_loop.get_tools(request)` to get available tools
2. Returns 202 Accepted with `InitResponse` containing tools
3. Starts the agent loop in a background task

### 2. Agent Loop Execution

The agent runs asynchronously after returning the 202 response:

```
RolloutServer                       TrainGate
    │                                   │
    │  POST /v1/chat/completions        │
    │  {rollout_id, messages, ...}      │
    ├──────────────────────────────────►│
    │                                   │
    │       LLM Response                │
    │  {choices, usage, token_ids}      │
    │◄──────────────────────────────────┤
    │                                   │
    │  (Execute tools locally)          │
    │                                   │
    │  POST /v1/chat/completions        │
    │  (with tool results appended)     │
    ├──────────────────────────────────►│
    │                                   │
    │  ... repeat until done ...        │
    │                                   │
```

Key points:
- RolloutServer calls TrainGate's `/v1/chat/completions` for LLM generation
- Messages are append-only (never modify previous messages)
- The `rollout_id` routes requests to the correct training session

### 3. Completion Notification

When the agent loop finishes:

```
RolloutServer                       TrainGate
    │                                   │
    │  POST /v1/rollout/completed       │
    │  {rollout_id, status,             │
    │   final_messages, metrics}        │
    ├──────────────────────────────────►│
    │                                   │
    │          200 OK                   │
    │◄──────────────────────────────────┤
    │                                   │
```

The `RolloutResponse` includes:
- `status`: "COMPLETED" or "ERROR"
- `final_messages`: Complete conversation history
- `finish_reason`: Why the rollout ended
- `metrics`: Execution statistics

## Core Components

### RolloutAgentLoop

The abstract base class that users implement:

```python
class RolloutAgentLoop(ABC):
    name: str  # Unique identifier

    @abstractmethod
    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        """Return tools for this rollout."""

    @abstractmethod
    async def run(self, ctx: RolloutContext) -> RolloutResult:
        """Execute the agent loop."""
```

The `__init_subclass__` hook validates that subclasses define `name`.

### RolloutContext

Execution context provided to the agent:

```python
@dataclass
class RolloutContext:
    request: RolloutRequest      # Original request
    tools: list[OpenAIFunctionToolSchema]      # Available tools
    llm: LLMClientProtocol       # LLM client (OsmosisLLMClient in production)

    async def chat(self, messages, **kwargs) -> CompletionsResult
    def complete(self, messages, finish_reason="stop", reward=None) -> RolloutResult
    def error(self, message, final_messages=None) -> RolloutResult
    def record_tool_call(self, latency_ms=0.0) -> None
```

### OsmosisLLMClient

HTTP client for TrainGate communication:

- Connection pooling (up to 100 connections)
- Automatic retry with exponential backoff (5xx errors, timeouts, transport errors)
- Timeout handling (default 300 seconds)
- Metrics collection (latency, token counts)

### AppState

Server state management:

- Tracks running and completed rollouts
- Provides idempotency (duplicate requests return same response)
- Background cleanup of old records
- Concurrency limiting via semaphore

## Error Handling

### Exception Hierarchy

```
OsmosisRolloutError (base)
├── OsmosisTransportError     # Network errors
├── OsmosisServerError        # 5xx errors (retryable)
├── OsmosisValidationError    # 4xx errors (not retryable)
├── OsmosisTimeoutError       # Timeouts
├── AgentLoopNotFoundError    # Registry lookup failed
├── ToolExecutionError        # Tool execution errors
└── ToolArgumentError         # Tool argument parsing errors
```

### Retry Strategy

| Error Type | Retried? | Notes |
|------------|----------|-------|
| 5xx Server Error | Yes | Exponential backoff (1s, 2s, 4s, ...) |
| 4xx Client Error | No | Fails immediately |
| Timeout | Yes | Up to max_retries |
| Network Error | Yes | Connection failures, DNS, etc. |

### Agent Error Handling

If an exception occurs in `agent_loop.run()`:

1. The error is logged
2. `complete_rollout()` is called with status="ERROR"
3. The error message is included in the response

## Metrics Collection

The SDK automatically tracks:

| Metric | Description |
|--------|-------------|
| `total_latency_ms` | Total rollout duration |
| `llm_latency_ms` | Time spent in LLM calls |
| `tool_latency_ms` | Time spent in tool execution |
| `num_llm_calls` | Number of LLM completion calls |
| `num_tool_calls` | Number of tool executions |
| `prompt_tokens` | Total prompt tokens |
| `response_tokens` | Total response tokens |

Use `ctx.record_tool_call(latency_ms)` to track tool execution time.

## Idempotency

The server handles duplicate requests:

1. Check if `rollout_id` is already running or recently completed
2. If duplicate, return the same `InitResponse` without starting new task
3. Completed rollout records are kept for `record_ttl_seconds` (default: 1 hour)

This ensures safety during network retries.

## Concurrency Control

The server limits concurrent rollouts via `max_concurrent` parameter:

```python
app = create_app(agent_loop, max_concurrent=100)
```

When the limit is reached, new rollouts wait for a slot to become available.

## Lifecycle Management

On server startup:
- Background cleanup task starts (prunes old completed records every 60s)

On server shutdown:
- Cleanup task is cancelled
- All running rollout tasks are cancelled
- Resources are released

## See Also

- [Remote Rollout Overview](./overview.md) -- Quick start guide
- [Agent Loop Guide](./agent-loop.md) -- Complete API documentation
- [Examples](./examples.md) -- Working code examples
