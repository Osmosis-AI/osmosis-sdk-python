# Agent Loop Guide

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

**Subclass Validation (`__init_subclass__`):**

When you define a concrete (non-abstract) subclass of `RolloutAgentLoop`, Python automatically validates at class definition time that the `name` class attribute is defined and non-empty. If it is missing or falsy, a `TypeError` is raised immediately:

```python
# This raises TypeError at class definition time:
class BadAgent(RolloutAgentLoop):
    # Missing 'name' attribute!
    def get_tools(self, request):
        return []
    async def run(self, ctx):
        return ctx.complete([])
# TypeError: Agent loop class BadAgent must define a 'name' class attribute
```

Abstract subclasses (those with remaining abstract methods) are not validated, so you can create intermediate base classes without defining `name`.

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

Return the default tool list for discovery and validation purposes, without requiring a real `RolloutRequest`.

Internally, this calls `get_tools()` with a synthetic request (`rollout_id="discovery"`, empty messages and params). This is used by the validation framework (`validate_agent_loop()`) and platform registration to discover the agent's tools without an active rollout.

```python
agent = MyAgent()
tools = agent.get_default_tools()
print(f"Agent provides {len(tools)} tools")
```

- **Returns:** list of `OpenAIFunctionToolSchema` objects
- **Notes:** If your `get_tools()` returns different tools based on request metadata, `get_default_tools()` returns the tools for an empty/default request.

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
| `llm` | `LLMClientProtocol` | LLM client for chat completions. In production (served via `create_app()` / `serve_agent_loop()`), this is an `OsmosisLLMClient` that calls back to TrainGate. In test/eval mode (e.g., `osmosis test`), this may be an `ExternalLLMClient` that routes requests to an external provider (OpenAI, Anthropic, etc.) via LiteLLM. |

**Methods:**

#### `async chat(messages, **kwargs) -> CompletionsResult`

Shorthand for `self.llm.chat_completions()`.

```python
result = await ctx.chat(messages, temperature=0.7)
```

#### `complete(final_messages, finish_reason="stop", reward=None) -> RolloutResult`

Create a successful completion result.

```python
return ctx.complete(messages, finish_reason="stop")

# With reward
return ctx.complete(messages, finish_reason="stop", reward=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `final_messages` | `List[Dict[str, Any]]` | required | Final conversation messages |
| `finish_reason` | `str` | `"stop"` | Why the rollout ended |
| `reward` | `float \| None` | `None` | Optional precomputed trajectory reward score |

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

#### `log_event(event_type, **data) -> None`

Log a debug event to the rollout's JSONL file. No-op if debug logging is not enabled.

```python
# Log before LLM call
ctx.log_event("pre_llm", turn=0, num_messages=len(messages))

# Log LLM response
ctx.log_event("llm_response", turn=0, has_tool_calls=result.has_tool_calls)

# Log tool results
ctx.log_event("tool_results", turn=0, num_results=len(tool_results))

# Log completion
ctx.log_event("rollout_complete", finish_reason="stop", reward=1.0)
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
    # api_key="... optional TrainGate bearer token ...",
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
| `api_key` | `Optional[str]` | `None` | Optional Bearer token attached to requests to TrainGate. In the remote-rollout protocol this value is provided by TrainGate in `RolloutRequest.api_key`, and RolloutServer forwards it on callbacks to TrainGate. |
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
- Extra/unknown keyword arguments are accepted and ignored.

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

# Start with validation (default)
serve_agent_loop(agent_loop, port=9000)

# Skip validation
serve_agent_loop(agent_loop, port=9000, validate=False)

# Full options
serve_agent_loop(
    agent_loop,
    host="0.0.0.0",
    port=9000,
    validate=True,
    log_level="info",
    reload=False,
    settings=None,
    skip_register=False,
    api_key=None,
    local_debug=False,
    debug_dir=None,
)
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
| `skip_register` | `bool` | `False` | Skip registering with Osmosis Platform (local testing mode) |
| `api_key` | `Optional[str]` | `None` | RolloutServer API key used by TrainGate to call this server via `Authorization: Bearer <api_key>` (generated if not provided). NOT related to `osmosis login`. |
| `local_debug` | `bool` | `False` | Local debug mode: disable API key auth and force `skip_register=True` (NOT for production) |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logging. If set, creates a timestamped subdirectory `{debug_dir}/{timestamp}/` and writes execution traces to `{rollout_id}.jsonl` files |

**Raises:**
- `ImportError`: If FastAPI or uvicorn not installed
- `AgentLoopValidationError`: If validation fails

---

### create_app()

Factory function to create a FastAPI application.

```python
from osmosis_ai.rollout import create_app

app = create_app(
    agent_loop,
    max_concurrent=100,
    record_ttl_seconds=3600,
    debug_dir="./rollout_logs",  # Optional: enable debug logging
)

# Full options
app = create_app(
    agent_loop,
    max_concurrent=100,
    record_ttl_seconds=3600,
    settings=None,
    credentials=my_credentials,         # For platform registration
    server_host="0.0.0.0",
    server_port=9000,
    api_key="my-secret-key",
    debug_dir="./rollout_logs",
    on_startup=my_startup_callback,
    on_shutdown=my_shutdown_callback,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Your agent implementation |
| `max_concurrent` | `int \| None` | `None` | Max concurrent rollouts (defaults to settings) |
| `record_ttl_seconds` | `float \| None` | `None` | TTL for completed records (defaults to settings) |
| `settings` | `Optional[RolloutSettings]` | `None` | Global settings override (server/logging/tracing) |
| `credentials` | `Optional[WorkspaceCredentials]` | `None` | Workspace credentials for platform registration. If `None`, registration is skipped |
| `server_host` | `Optional[str]` | `None` | Host the server is bound to (used for platform registration) |
| `server_port` | `Optional[int]` | `None` | Port the server is listening on (used for platform registration) |
| `api_key` | `Optional[str]` | `None` | API key for authenticating incoming requests. If provided, requests must include `Authorization: Bearer <api_key>` |
| `debug_dir` | `Optional[str]` | `None` | Directory for debug logging. If set, each rollout writes execution traces to `{debug_dir}/{rollout_id}.jsonl`. Note: when using `serve_agent_loop()` or CLI, a timestamped subdirectory is created automatically |
| `on_startup` | `Optional[Callable[[], Awaitable[None]]]` | `None` | Async callback to run during application startup (e.g., warming caches, starting background services) |
| `on_shutdown` | `Optional[Callable[[], Awaitable[None]]]` | `None` | Async callback to run during application shutdown (e.g., stopping services, releasing resources) |

**Returns:** `FastAPI` application

**Generated Endpoints:**

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/v1/rollout/init` | POST | 202 | Accept rollout request |
| `/health` | GET | 200 | Health check (unauthenticated) |
| `/platform/health` | GET | 200 | Authenticated health check for Osmosis Platform. Requires `Authorization: Bearer <api_key>`. Returns 404 if `api_key` is not configured (e.g., `local_debug` mode). Used by the platform to verify reachability and API key correctness. |

---

## Validation

### validate_agent_loop()

Validate a RolloutAgentLoop implementation.

```python
from osmosis_ai.rollout import validate_agent_loop

result = validate_agent_loop(agent_loop)

if result.valid:
    print(f"Agent '{result.agent_name}' is valid with {result.tool_count} tools")
else:
    for error in result.errors:
        print(f"Error: {error}")

# Or raise exception if invalid
result.raise_if_invalid()
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_loop` | `RolloutAgentLoop` | required | Agent loop to validate |
| `request` | `Optional[RolloutRequest]` | `None` | Custom request for testing get_tools() |

**Returns:** `ValidationResult`

**Validation Checks:**
- `name` attribute is defined and non-empty
- `get_tools()` returns a valid list
- Each tool conforms to OpenAI function schema
- `run()` method is async

---

### ValidationResult

Result of agent loop validation.

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    agent_name: Optional[str]
    tool_count: int
```

**Methods:**

#### `raise_if_invalid() -> None`

Raise `AgentLoopValidationError` if validation failed.

#### `__bool__() -> bool`

Returns `True` if validation passed.

```python
result = validate_agent_loop(agent_loop)
if result:  # Same as result.valid
    print("Valid!")
```

---

### ValidationError

A single validation error or warning.

```python
@dataclass
class ValidationError:
    code: str        # e.g., "MISSING_NAME", "INVALID_TOOL_TYPE"
    message: str     # Human-readable message
    field: Optional[str]  # Field that caused error
    details: Optional[Dict[str, Any]]  # Additional context
```

**Common Error Codes:**

| Code | Description |
|------|-------------|
| `MISSING_NAME` | Agent loop has no `name` attribute |
| `EMPTY_NAME` | `name` is empty or whitespace |
| `GET_TOOLS_EXCEPTION` | `get_tools()` raised an exception |
| `GET_TOOLS_RETURNS_NONE` | `get_tools()` returned `None` |
| `MISSING_TOOL_TYPE` | Tool missing `type` field |
| `MISSING_FUNCTION` | Tool missing `function` field |
| `MISSING_FUNCTION_NAME` | Function missing `name` field |
| `RUN_NOT_ASYNC` | `run()` method is not async |

---

### AgentLoopValidationError

Exception raised when validation fails.

```python
from osmosis_ai.rollout import AgentLoopValidationError

try:
    result.raise_if_invalid()
except AgentLoopValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.errors:
        print(f"  - {error}")
```

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
    api_key: Optional[str] = None  # Optional Bearer token RolloutServer uses for callbacks to TrainGate
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

## See Also

- [Remote Rollout Overview](./overview.md) - Quick start guide
- [Architecture](./architecture.md) - Protocol design
- [Examples](./examples.md) - Working code examples
