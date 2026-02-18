# Agent Loop Guide

Complete API documentation for the Osmosis Remote Rollout SDK.

## Core Classes

### RolloutAgentLoop

Abstract base class for agent loop implementations.

```python
from osmosis_ai.rollout import RolloutAgentLoop

class CalculatorAgentLoop(RolloutAgentLoop):
    name = "calculator"  # Required class attribute

    def get_tools(self, request: RolloutRequest) -> list[OpenAIFunctionToolSchema]:
        ...

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        ...
```

**Class Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for this agent loop (required) |

Concrete subclasses must define `name`; omitting it raises `TypeError` at class definition time.

**Abstract Methods:**

#### `get_tools(request: RolloutRequest) -> list[OpenAIFunctionToolSchema]`

Return tools available for this rollout. Can return different tools based on `request.metadata`.

#### `async run(ctx: RolloutContext) -> RolloutResult`

Execute the agent loop. Messages must be append-only; never modify previous messages.

**Convenience Methods:**

#### `get_default_tools() -> list[OpenAIFunctionToolSchema]`

Return the default tool list for discovery and validation. Calls `get_tools()` with a synthetic request.

---

### RolloutContext

Execution context provided to the agent loop.

```python
@dataclass
class RolloutContext:
    request: RolloutRequest
    tools: list[OpenAIFunctionToolSchema]
    llm: LLMClientProtocol
```

**Methods:**

#### `async chat(messages, **kwargs) -> CompletionsResult`

Shorthand for `self.llm.chat_completions()`.

#### `complete(final_messages, finish_reason="stop", reward=None) -> RolloutResult`

Create a successful completion result.

#### `error(error_message, final_messages=None) -> RolloutResult`

Create an error result.

#### `record_tool_call(latency_ms=0.0) -> None`

Record a tool call for metrics.

#### `log_event(event_type, **data) -> None`

Log a debug event to the rollout's JSONL file. No-op if debug logging is not enabled.

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
    reward: Optional[float] = None           # Precomputed trajectory reward
```

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

### serve_agent_loop()

Start a RolloutServer with automatic validation.

```python
from osmosis_ai.rollout import serve_agent_loop

serve_agent_loop(agent_loop, port=9000)
```

### create_app()

Factory function to create a FastAPI application.

```python
from osmosis_ai.rollout import create_app

app = create_app(agent_loop, max_concurrent=100, record_ttl_seconds=3600)
```

See docstrings for full parameter lists. Generated endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/rollout/init` | POST | Accept rollout request (returns 202) |
| `/health` | GET | Health check |
| `/platform/health` | GET | Authenticated health check for Osmosis Platform |

---

## Validation

### validate_agent_loop()

```python
from osmosis_ai.rollout import validate_agent_loop

result = validate_agent_loop(agent_loop)
if not result.valid:
    result.raise_if_invalid()
```

Returns `ValidationResult` with `valid`, `errors`, `warnings`, `agent_name`, `tool_count`.

---

## Schemas

### RolloutRequest

```python
class RolloutRequest(BaseModel):
    rollout_id: str              # 1-256 chars, not empty
    server_url: str              # TrainGate URL (http/https)
    messages: List[MessageDict]  # Initial conversation
    completion_params: SamplingParamsDict
    tool_server_url: Optional[str] = None
    max_turns: int = 10
    max_tokens_total: int = 8192
    metadata: Dict[str, Any] = {}
    api_key: Optional[str] = None
    idempotency_key: Optional[str] = None
```

### InitResponse

```python
class InitResponse(BaseModel):
    rollout_id: str
    tools: List[OpenAIFunctionToolSchema] = []
```

### RolloutResponse

```python
class RolloutResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus        # COMPLETED or ERROR
    final_messages: List[MessageDict] = []
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    reward: Optional[float] = None
    metrics: Optional[RolloutMetrics] = None
    extra_fields: Dict[str, Any] = {}
```

### OpenAIFunctionToolSchema

```python
class OpenAIFunctionToolSchema(BaseModel):
    type: str  # "function"
    function: OpenAIFunctionSchema

class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False
```

### RolloutMetrics

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

---

## Exceptions

All exceptions inherit from `OsmosisRolloutError`.

| Exception | Description |
|-----------|-------------|
| `OsmosisRolloutError` | Base exception for all rollout errors |
| `OsmosisTransportError` | Network/transport level errors |
| `OsmosisServerError` | Server returned 5xx (retryable). Has `status_code`. |
| `OsmosisValidationError` | Server returned 4xx (not retryable). Has `status_code`. |
| `OsmosisTimeoutError` | Request timed out |
| `AgentLoopNotFoundError` | Agent loop not found in registry |
| `AgentLoopValidationError` | Validation failed. Has `errors` attribute. |
| `ToolExecutionError` | Tool execution failed |
| `ToolArgumentError` | Tool argument parsing failed (inherits `ToolExecutionError`) |

## See Also

- [Architecture](./architecture.md) - Protocol design
- [Examples](./examples.md) - Working code examples
