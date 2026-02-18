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

Concrete subclasses must define `name`; omitting it raises `TypeError` at class definition time. Abstract subclasses (with remaining abstract methods) are not validated, so intermediate base classes don't need `name`.

**Abstract Methods:**

#### `get_tools(request: RolloutRequest) -> list[OpenAIFunctionToolSchema]`

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

**Convenience Methods:**

#### `get_default_tools() -> list[OpenAIFunctionToolSchema]`

Return the default tool list for discovery and validation purposes, without requiring a real `RolloutRequest`. Calls `get_tools()` with a synthetic request. Used by the validation framework and platform registration.

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

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `request` | `RolloutRequest` | Original rollout request |
| `tools` | `list[OpenAIFunctionToolSchema]` | Tools returned by `get_tools()` |
| `llm` | `LLMClientProtocol` | LLM client (production: `OsmosisLLMClient` calling TrainGate; test/eval: `ExternalLLMClient` via LiteLLM) |

**Methods:**

#### `async chat(messages, **kwargs) -> CompletionsResult`

Shorthand for `self.llm.chat_completions()`.

```python
result = await ctx.chat(messages, temperature=0.7)
```

#### `complete(final_messages, finish_reason="stop", reward=None) -> RolloutResult`

Create a successful completion result.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `final_messages` | `List[Dict[str, Any]]` | required | Final conversation messages |
| `finish_reason` | `str` | `"stop"` | Why the rollout ended |
| `reward` | `float \| None` | `None` | Optional precomputed trajectory reward score |

#### `error(error_message, final_messages=None) -> RolloutResult`

Create an error result.

#### `record_tool_call(latency_ms=0.0) -> None`

Record a tool call for metrics.

#### `log_event(event_type, **data) -> None`

Log a debug event to the rollout's JSONL file. No-op if debug logging is not enabled.

```python
ctx.log_event("pre_llm", turn=0, num_messages=len(messages))
ctx.log_event("llm_response", turn=0, has_tool_calls=result.has_tool_calls)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `debug_enabled` | `bool` | Whether debug logging is enabled |

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

## Client

### OsmosisLLMClient

HTTP client for calling TrainGate's LLM completion endpoint.

```python
from osmosis_ai.rollout import OsmosisLLMClient

async with OsmosisLLMClient(
    server_url="http://trainer:8080",
    rollout_id="rollout-123",
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
| `api_key` | `Optional[str]` | `None` | Bearer token for TrainGate callbacks (provided by TrainGate in `RolloutRequest.api_key`) |
| `timeout_seconds` | `float` | `300.0` | Request timeout (seconds) |
| `max_retries` | `int` | `3` | Max retries for transient errors (5xx / timeouts / transport) |
| `complete_rollout_retries` | `int` | `2` | Max retries for `/v1/rollout/completed` callback |
| `settings` | `Optional[RolloutClientSettings]` | `None` | Advanced client settings |

**Methods:**

#### `async chat_completions(messages, **kwargs) -> CompletionsResult`

Call TrainGate's `/v1/chat/completions` endpoint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict]` | required | Conversation history |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Top-p sampling |
| `max_tokens` | `int` | `512` | Maximum response tokens |
| `stop` | `Optional[Union[str, List[str]]]` | `None` | Stop sequences |
| `logprobs` | `bool` | `True` | Return log probabilities |

Extra/unknown keyword arguments are accepted and ignored.

#### `async complete_rollout(status, final_messages, ...) -> None`

Notify TrainGate that rollout is complete.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `str` | required | "COMPLETED" or "ERROR" |
| `final_messages` | `List[Dict]` | required | Final conversation |
| `finish_reason` | `str` | `"stop"` | Why rollout ended |
| `error_message` | `Optional[str]` | `None` | Error description |
| `metrics` | `Optional[RolloutMetrics]` | `None` | Execution metrics |
| `reward` | `Optional[float]` | `None` | Precomputed trajectory reward score |

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

### serve_agent_loop()

Start a RolloutServer with automatic validation.

```python
from osmosis_ai.rollout import serve_agent_loop

serve_agent_loop(agent_loop, port=9000)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Your agent implementation |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |
| `port` | `int` | `9000` | Port to bind to |
| `validate` | `bool` | `True` | Validate agent loop before starting |
| `log_level` | `str` | `"info"` | Uvicorn log level |
| `reload` | `bool` | `False` | Enable auto-reload for development |
| `settings` | `Optional[RolloutSettings]` | `None` | Configuration override |
| `skip_register` | `bool` | `False` | Skip registering with Osmosis Platform |
| `api_key` | `Optional[str]` | `None` | RolloutServer API key used by TrainGate (generated if not provided) |
| `local_debug` | `bool` | `False` | Disable API key auth and force `skip_register=True` |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logging |

**Raises:** `ImportError` (missing FastAPI/uvicorn), `AgentLoopValidationError` (validation fails)

---

### create_app()

Factory function to create a FastAPI application.

```python
from osmosis_ai.rollout import create_app

app = create_app(agent_loop, max_concurrent=100, record_ttl_seconds=3600)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Your agent implementation |
| `max_concurrent` | `int \| None` | `None` | Max concurrent rollouts |
| `record_ttl_seconds` | `float \| None` | `None` | TTL for completed records |
| `settings` | `Optional[RolloutSettings]` | `None` | Global settings override |
| `credentials` | `Optional[WorkspaceCredentials]` | `None` | Workspace credentials for platform registration |
| `server_host` | `Optional[str]` | `None` | Host (for platform registration) |
| `server_port` | `Optional[int]` | `None` | Port (for platform registration) |
| `api_key` | `Optional[str]` | `None` | API key for authenticating incoming requests |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logging |
| `on_startup` | `Optional[Callable[[], Awaitable[None]]]` | `None` | Async startup callback |
| `on_shutdown` | `Optional[Callable[[], Awaitable[None]]]` | `None` | Async shutdown callback |

**Generated Endpoints:**

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/v1/rollout/init` | POST | 202 | Accept rollout request |
| `/health` | GET | 200 | Health check (unauthenticated) |
| `/platform/health` | GET | 200 | Authenticated health check for Osmosis Platform |

---

## Validation

### validate_agent_loop()

Validate a RolloutAgentLoop implementation.

```python
from osmosis_ai.rollout import validate_agent_loop

result = validate_agent_loop(agent_loop)
if not result.valid:
    result.raise_if_invalid()
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Agent loop to validate |
| `request` | `Optional[RolloutRequest]` | `None` | Custom request for testing get_tools() |

**Returns:** `ValidationResult` with `valid`, `errors`, `warnings`, `agent_name`, `tool_count`.

**Validation Checks:**
- `name` attribute is defined and non-empty
- `get_tools()` returns a valid list
- Each tool conforms to OpenAI function schema
- `run()` method is async

**Common Error Codes:** `MISSING_NAME`, `EMPTY_NAME`, `GET_TOOLS_EXCEPTION`, `GET_TOOLS_RETURNS_NONE`, `MISSING_TOOL_TYPE`, `MISSING_FUNCTION`, `MISSING_FUNCTION_NAME`, `RUN_NOT_ASYNC`

---

## Registry

Functions for managing multiple agent loop instances:

| Function | Description |
|----------|-------------|
| `register_agent_loop(instance)` | Register an agent loop. Raises `ValueError` if name is already registered. |
| `get_agent_loop(name)` | Get by name. Raises `AgentLoopNotFoundError` if not found. |
| `list_agent_loops()` | List all registered agent loop names. |
| `unregister_agent_loop(name)` | Remove from registry. Returns `True` if found. |

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
    api_key: Optional[str] = None  # Optional Bearer token for callbacks to TrainGate
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
    reward: Optional[float] = None   # Precomputed trajectory reward
    metrics: Optional[RolloutMetrics] = None
    extra_fields: Dict[str, Any] = {}
```

### OpenAIFunctionToolSchema

OpenAI-compatible tool definition.

```python
class OpenAIFunctionToolSchema(BaseModel):
    type: str  # "function"
    function: OpenAIFunctionSchema

class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema  # defaults to empty object
    strict: bool = False

class OpenAIFunctionParametersSchema(BaseModel):
    type: str  # usually "object"
    properties: Dict[str, OpenAIFunctionPropertySchema]
    required: List[str]

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

```python
class RolloutStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
```

---

## Exceptions

All exceptions inherit from `OsmosisRolloutError`.

| Exception | Description |
|-----------|-------------|
| `OsmosisRolloutError` | Base exception for all rollout errors |
| `OsmosisTransportError` | Network/transport level errors |
| `OsmosisServerError` | Server returned 5xx (retryable). Has `status_code` attribute. |
| `OsmosisValidationError` | Server returned 4xx (not retryable). Has `status_code` attribute. |
| `OsmosisTimeoutError` | Request timed out |
| `AgentLoopNotFoundError` | Agent loop not found in registry. Has `name` and `available` attributes. |
| `AgentLoopValidationError` | Validation failed. Has `errors` attribute. |
| `ToolExecutionError` | Tool execution failed. Has optional `tool_call_id` and `tool_name`. |
| `ToolArgumentError` | Tool argument parsing failed (inherits from `ToolExecutionError`). |

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

## See Also

- [Remote Rollout Overview](./overview.md) - Quick start guide
- [Architecture](./architecture.md) - Protocol design
- [Examples](./examples.md) - Working code examples
