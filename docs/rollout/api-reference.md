# API Reference

Complete API documentation for the Osmosis Remote Rollout SDK.

## Core Classes

### RolloutAgentLoop

Abstract base class for agent loop implementations.

```python
from osmosis_ai.rollout import RolloutAgentLoop

class MyAgent(RolloutAgentLoop):
    name = "my_agent"  # Required class attribute

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        ...

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        ...
```

**Class Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for this agent loop (required) |

**Abstract Methods:**

#### `get_tools(request: RolloutRequest) -> List[OpenAIFunctionToolSchema]`

Return tools available for this rollout.

- **Parameters:**
  - `request`: The incoming rollout request
- **Returns:** List of `OpenAIFunctionToolSchema` objects
- **Notes:** Can return different tools based on `request.metadata`

#### `async run(ctx: RolloutContext) -> RolloutResult`

Execute the agent loop.

- **Parameters:**
  - `ctx`: Execution context with LLM client and request
- **Returns:** `RolloutResult` with final status and messages
- **Notes:** Messages must be append-only; never modify previous messages

---

### RolloutContext

Execution context provided to the agent loop.

```python
@dataclass
class RolloutContext:
    request: RolloutRequest
    tools: List[OpenAIFunctionToolSchema]
    llm: OsmosisLLMClient
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `request` | `RolloutRequest` | Original rollout request |
| `tools` | `List[OpenAIFunctionToolSchema]` | Tools returned by `get_tools()` |
| `llm` | `OsmosisLLMClient` | HTTP client for LLM calls |

**Methods:**

#### `async chat(messages, **kwargs) -> CompletionsResult`

Shorthand for `self.llm.chat_completions()`.

```python
result = await ctx.chat(messages, temperature=0.7)
```

#### `complete(final_messages, finish_reason="stop") -> RolloutResult`

Create a successful completion result.

```python
return ctx.complete(messages, finish_reason="stop")
```

#### `error(error_message, final_messages=None) -> RolloutResult`

Create an error result.

```python
return ctx.error("Tool execution failed", final_messages=messages)
```

#### `record_tool_call(latency_ms=0.0) -> None`

Record a tool call for metrics.

```python
start = time.monotonic()
result = execute_tool(...)
ctx.record_tool_call(latency_ms=(time.monotonic() - start) * 1000)
```

---

### RolloutResult

Result returned from agent loop execution.

```python
@dataclass
class RolloutResult:
    status: str                              # "COMPLETED" or "ERROR"
    final_messages: List[Dict[str, Any]]
    finish_reason: str
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None
```

---

## Client

### OsmosisLLMClient

HTTP client for calling TrainGate's LLM completion endpoint.

```python
from osmosis_ai.rollout import OsmosisLLMClient

async with OsmosisLLMClient(
    server_url="http://trainer:8080",
    rollout_id="rollout-123",
    api_key="optional-token",
    timeout_seconds=300.0,
    max_retries=3,
) as client:
    result = await client.chat_completions(messages)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | required | TrainGate base URL |
| `rollout_id` | `str` | required | Unique rollout identifier |
| `api_key` | `Optional[str]` | `None` | Bearer token for authentication |
| `timeout_seconds` | `float` | `300.0` | Request timeout (seconds) |
| `max_retries` | `int` | `3` | Max retries for transient errors (5xx / timeouts / transport) |
| `complete_rollout_retries` | `int` | `2` | Max retries for `/v1/rollout/completed` callback |
| `settings` | `Optional[RolloutClientSettings]` | `None` | Advanced client settings (pool/retry delays); defaults to global settings |

**Methods:**

#### `async chat_completions(messages, **kwargs) -> CompletionsResult`

Call TrainGate's `/v1/chat/completions` endpoint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict]` | required | Conversation history |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Top-p sampling |
| `max_tokens` | `int` | `512` | Maximum response tokens |
| `stop` | `Optional[Union[str, List[str]]]` | `None` | Stop sequences (string is normalized to list) |
| `logprobs` | `bool` | `True` | Return log probabilities |

Notes:
- Extra/unknown keyword arguments are accepted and ignored for forward compatibility.

**Raises:**
- `OsmosisTransportError`: Network errors
- `OsmosisServerError`: 5xx errors (after retries exhausted)
- `OsmosisValidationError`: 4xx errors
- `OsmosisTimeoutError`: Request timeout

#### `async complete_rollout(status, final_messages, ...) -> None`

Notify TrainGate that rollout is complete.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `str` | required | "COMPLETED" or "ERROR" |
| `final_messages` | `List[Dict]` | required | Final conversation |
| `finish_reason` | `str` | `"stop"` | Why rollout ended |
| `error_message` | `Optional[str]` | `None` | Error description |
| `metrics` | `Optional[RolloutMetrics]` | `None` | Execution metrics |

#### `get_metrics() -> RolloutMetrics`

Get current metrics from this client session.

---

### CompletionsResult

Result from LLM completion call.

```python
@dataclass(frozen=True)
class CompletionsResult:
    message: Dict[str, Any]    # Assistant message
    token_ids: List[int]       # For training
    logprobs: List[float]      # For training
    usage: Dict[str, int]      # Token statistics
    finish_reason: str
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `has_tool_calls` | `bool` | Whether response contains tool calls |
| `tool_calls` | `List[Dict]` | List of tool call objects |
| `content` | `Optional[str]` | Text content of response |

---

## Server

### create_app()

Factory function to create a FastAPI application.

```python
from osmosis_ai.rollout import create_app

app = create_app(
    agent_loop,
    max_concurrent=100,
    record_ttl_seconds=3600,
    enable_metrics_endpoint=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Your agent implementation |
| `max_concurrent` | `int` | `100` | Max concurrent rollouts |
| `record_ttl_seconds` | `float` | `3600` | TTL for completed records |
| `settings` | `Optional[RolloutSettings]` | `None` | Global settings override (server/logging/tracing/metrics) |
| `enable_metrics_endpoint` | `bool` | `True` | Expose `/metrics` when metrics are enabled |

**Returns:** `FastAPI` application

**Generated Endpoints:**

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/v1/rollout/init` | POST | 202 | Accept rollout request |
| `/health` | GET | 200 | Health check |
| `/metrics` | GET | 200 | Prometheus metrics (when enabled) |

---

## Registry

### register_agent_loop()

Register an agent loop with the global registry.

```python
from osmosis_ai.rollout import register_agent_loop

register_agent_loop(MyAgentLoop())
```

**Raises:** `ValueError` if name already registered

### get_agent_loop()

Get an agent loop from the global registry.

```python
from osmosis_ai.rollout import get_agent_loop

loop = get_agent_loop("my_agent")
```

**Raises:** `AgentLoopNotFoundError` if not found

### list_agent_loops()

List all registered agent loop names.

```python
from osmosis_ai.rollout import list_agent_loops

names = list_agent_loops()  # ["agent1", "agent2"]
```

### unregister_agent_loop()

Remove an agent loop from the registry.

```python
from osmosis_ai.rollout import unregister_agent_loop

success = unregister_agent_loop("my_agent")  # True or False
```

---

## Schemas

### RolloutRequest

Request sent to `/v1/rollout/init` to start a rollout.

```python
class RolloutRequest(BaseModel):
    rollout_id: str              # 1-256 chars, not empty
    server_url: str              # TrainGate URL (http/https)
    messages: List[MessageDict]  # Initial conversation
    completion_params: SamplingParamsDict
    tool_server_url: Optional[str] = None
    max_turns: int = 10
    max_tokens_total: int = 8192
    metadata: Dict[str, Any] = {}  # JSON-serializable (size-limited; default 1MB)
    api_key: Optional[str] = None
    idempotency_key: Optional[str] = None
```

### InitResponse

Response from `/v1/rollout/init` endpoint.

```python
class InitResponse(BaseModel):
    rollout_id: str
    tools: List[OpenAIFunctionToolSchema] = []
```

### RolloutResponse

Posted to `/v1/rollout/completed`.

```python
class RolloutResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus        # COMPLETED or ERROR
    final_messages: List[MessageDict] = []
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None
    extra_fields: Dict[str, Any] = {}
```

### OpenAIFunctionToolSchema

OpenAI-compatible tool definition.

```python
class OpenAIFunctionToolSchema(BaseModel):
    type: str  # "function"
    function: OpenAIFunctionSchema
```

### OpenAIFunctionSchema

Function definition within a tool.

```python
class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = OpenAIFunctionParametersSchema(
        type="object",
        properties={},
        required=[],
    )
    strict: bool = False
```

### OpenAIFunctionParametersSchema

JSON Schema for function parameters.

```python
class OpenAIFunctionParametersSchema(BaseModel):
    type: str  # usually "object"
    properties: Dict[str, OpenAIFunctionPropertySchema]
    required: List[str]
```

### OpenAIFunctionPropertySchema

Single parameter definition.

```python
class OpenAIFunctionPropertySchema(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
```

### RolloutMetrics

Execution metrics.

```python
class RolloutMetrics(BaseModel):
    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    max_context_tokens: int = 0
```

### RolloutStatus

Enum for rollout status.

```python
class RolloutStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
```

---

## Exceptions

All exceptions inherit from `OsmosisRolloutError`.

### OsmosisRolloutError

Base exception for all rollout errors.

```python
try:
    await client.chat_completions(messages)
except OsmosisRolloutError as e:
    print(f"Rollout error: {e}")
```

### OsmosisTransportError

Network/transport level errors.

```python
# Connection failed, DNS resolution failed, etc.
```

### OsmosisServerError

Server returned 5xx error (retryable).

```python
except OsmosisServerError as e:
    print(f"Server error {e.status_code}: {e}")
```

### OsmosisValidationError

Server returned 4xx error (not retryable).

```python
except OsmosisValidationError as e:
    print(f"Validation error {e.status_code}: {e}")
```

### OsmosisTimeoutError

Request timed out.

```python
except OsmosisTimeoutError as e:
    print(f"Timeout: {e}")
```

### AgentLoopNotFoundError

Agent loop not found in registry.

```python
except AgentLoopNotFoundError as e:
    print(f"Agent '{e.name}' not found. Available: {e.available}")
```

### ToolExecutionError

Tool execution failed (optionally includes `tool_call_id` and `tool_name`).

```python
from osmosis_ai.rollout import ToolExecutionError

raise ToolExecutionError("Tool failed", tool_call_id="call_123", tool_name="add")
```

### ToolArgumentError

Tool argument parsing failed (inherits from `ToolExecutionError`).

```python
from osmosis_ai.rollout import ToolArgumentError

raise ToolArgumentError("Invalid JSON", tool_call_id="call_123", tool_name="add")
```

---

## Type Aliases

### MessageDict

```python
MessageDict = Dict[str, Any]
# Example: {"role": "user", "content": "Hello"}
# Example: {"role": "tool", "content": "42", "tool_call_id": "call_123"}
```

### SamplingParamsDict

```python
SamplingParamsDict = Dict[str, Any]
# Example: {"temperature": 0.7, "max_tokens": 512}
```
