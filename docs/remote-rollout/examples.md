# Examples

Complete working examples for the Osmosis Remote Rollout SDK.

## CLI Quick Start

The fastest way to run a RolloutServer is using the CLI:

```bash
# Validate agent loop (checks tools, async run, etc.)
osmosis validate -m my_agent:agent_loop

# Start server with validation (default port 9000)
osmosis serve -m my_agent:agent_loop

# Specify port
osmosis serve -m my_agent:agent_loop -p 8080

# Skip validation (not recommended)
osmosis serve -m my_agent:agent_loop --no-validate

# Enable auto-reload for development
osmosis serve -m my_agent:agent_loop --reload

# Enable debug logging (writes {rollout_id}.jsonl files to ./logs/)
osmosis serve -m my_agent:agent_loop --log ./logs

# Verbose validation output
osmosis validate -m my_agent:agent_loop -v
```

The module path format is `module:attribute`. The CLI automatically adds the current directory to Python path.

---

## Basic Calculator Agent

A simple agent that can perform arithmetic operations.

```python
# calculator_agent.py

import json
from typing import List

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    create_app,
)


class CalculatorAgent(RolloutAgentLoop):
    """Agent that can perform basic arithmetic."""

    name = "calculator"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        """Define the calculator tool."""
        return [
            OpenAIFunctionToolSchema(
                type="function",
                function=OpenAIFunctionSchema(
                    name="calculate",
                    description="Perform arithmetic calculation",
                    parameters=OpenAIFunctionParametersSchema(
                        properties={
                            "operation": OpenAIFunctionPropertySchema(
                                type="string",
                                description="The operation to perform",
                                enum=["add", "subtract", "multiply", "divide"],
                            ),
                            "a": OpenAIFunctionPropertySchema(
                                type="number",
                                description="First operand",
                            ),
                            "b": OpenAIFunctionPropertySchema(
                                type="number",
                                description="Second operand",
                            ),
                        },
                        required=["operation", "a", "b"],
                    ),
                ),
            ),
        ]

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool and return the result."""
        if name == "calculate":
            op = args["operation"]
            a, b = args["a"], args["b"]

            if op == "add":
                result = a + b
            elif op == "subtract":
                result = a - b
            elif op == "multiply":
                result = a * b
            elif op == "divide":
                if b == 0:
                    return "Error: Division by zero"
                result = a / b
            else:
                return f"Error: Unknown operation {op}"

            return str(result)

        return f"Error: Unknown tool {name}"

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        """Execute the agent loop."""
        messages = list(ctx.request.messages)

        for turn in range(ctx.request.max_turns):
            # Call LLM
            result = await ctx.chat(
                messages,
                **ctx.request.completion_params,
            )
            messages.append(result.message)

            # Check if done
            if not result.has_tool_calls:
                return ctx.complete(messages)

            # Process tool calls
            for tool_call in result.tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                # Execute tool
                tool_result = self.execute_tool(tool_name, tool_args)
                ctx.record_tool_call()

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                })

        return ctx.complete(messages, finish_reason="max_turns")


# Export instance for CLI usage
agent_loop = CalculatorAgent()

# Create FastAPI app
app = create_app(agent_loop)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

Run with CLI (recommended):
```bash
# Validate first
osmosis validate -m calculator_agent:agent_loop

# Start server
osmosis serve -m calculator_agent:agent_loop -p 9000
```

Or with uvicorn directly:
```bash
uvicorn calculator_agent:app --port 9000
```

---

## Multi-Tool Agent

An agent with multiple tools and dynamic tool selection.

```python
# multi_tool_agent.py

import json
from typing import List

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    create_app,
)


# Tool definitions
SEARCH_TOOL = OpenAIFunctionToolSchema(
    type="function",
    function=OpenAIFunctionSchema(
        name="search",
        description="Search for information on a topic",
        parameters=OpenAIFunctionParametersSchema(
            properties={
                "query": OpenAIFunctionPropertySchema(
                    type="string",
                    description="Search query",
                ),
            },
            required=["query"],
        ),
    ),
)

WEATHER_TOOL = OpenAIFunctionToolSchema(
    type="function",
    function=OpenAIFunctionSchema(
        name="get_weather",
        description="Get current weather for a location",
        parameters=OpenAIFunctionParametersSchema(
            properties={
                "location": OpenAIFunctionPropertySchema(
                    type="string",
                    description="City name",
                ),
            },
            required=["location"],
        ),
    ),
)

CALCULATOR_TOOL = OpenAIFunctionToolSchema(
    type="function",
    function=OpenAIFunctionSchema(
        name="calculate",
        description="Perform arithmetic",
        parameters=OpenAIFunctionParametersSchema(
            properties={
                "expression": OpenAIFunctionPropertySchema(
                    type="string",
                    description="Math expression to evaluate",
                ),
            },
            required=["expression"],
        ),
    ),
)


class MultiToolAgent(RolloutAgentLoop):
    """Agent with multiple tools."""

    name = "multi_tool_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        """Return tools based on metadata configuration."""
        # Check if specific tools are requested in metadata
        requested_tools = request.metadata.get("enabled_tools")

        if requested_tools:
            tools = []
            tool_map = {
                "search": SEARCH_TOOL,
                "weather": WEATHER_TOOL,
                "calculator": CALCULATOR_TOOL,
            }
            for name in requested_tools:
                if name in tool_map:
                    tools.append(tool_map[name])
            return tools

        # Default: return all tools
        return [SEARCH_TOOL, WEATHER_TOOL, CALCULATOR_TOOL]

    async def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool (mock implementations)."""
        if name == "search":
            return f"Search results for '{args['query']}': [Mock results]"

        if name == "get_weather":
            return f"Weather in {args['location']}: 72°F, Sunny"

        if name == "calculate":
            try:
                # WARNING: eval is unsafe in production!
                # Use a proper math parser instead
                result = eval(args["expression"])
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        return f"Unknown tool: {name}"

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        """Execute the agent loop."""
        messages = list(ctx.request.messages)

        for _ in range(ctx.request.max_turns):
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

            if not result.has_tool_calls:
                return ctx.complete(messages)

            for tool_call in result.tool_calls:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                tool_result = await self.execute_tool(name, args)
                ctx.record_tool_call()

                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                })

        return ctx.complete(messages, finish_reason="max_turns")


app = create_app(MultiToolAgent())
```

---

## Agent with Error Handling

Robust error handling in the agent loop.

```python
# robust_agent.py

import json
import logging
from typing import List

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
    OsmosisRolloutError,
    create_app,
)

logger = logging.getLogger(__name__)


class RobustAgent(RolloutAgentLoop):
    """Agent with comprehensive error handling."""

    name = "robust_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)

        for turn in range(ctx.request.max_turns):
            try:
                result = await ctx.chat(
                    messages,
                    **ctx.request.completion_params,
                )
                messages.append(result.message)

                if not result.has_tool_calls:
                    return ctx.complete(messages)

                # Process tools with individual error handling
                for tool_call in result.tool_calls:
                    try:
                        tool_result = await self.execute_tool_safe(tool_call)
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call["id"],
                        })
                        ctx.record_tool_call()
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        # Return error message to LLM so it can recover
                        messages.append({
                            "role": "tool",
                            "content": f"Error: {str(e)}",
                            "tool_call_id": tool_call["id"],
                        })

            except OsmosisRolloutError as e:
                # Network/server errors - return error result
                logger.error(f"LLM call failed: {e}")
                return ctx.error(
                    f"LLM call failed: {e}",
                    final_messages=messages,
                )

        return ctx.complete(messages, finish_reason="max_turns")

    async def execute_tool_safe(self, tool_call: dict) -> str:
        """Execute tool with timeout and error handling."""
        import asyncio

        async def execute():
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            # Your tool execution logic here
            return f"Result for {name}"

        try:
            return await asyncio.wait_for(execute(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Tool execution timed out")


app = create_app(RobustAgent())
```

---

## Agent with Debug Logging

Using `ctx.log_event()` to track execution for debugging and analysis.

```python
# logging_agent.py

import json
from typing import List

from osmosis_ai.rollout import (
    RolloutAgentLoop,
    RolloutContext,
    RolloutRequest,
    RolloutResult,
    OpenAIFunctionToolSchema,
    create_app,
)


class LoggingAgent(RolloutAgentLoop):
    """Agent that logs execution details for debugging."""

    name = "logging_agent"

    def get_tools(self, request: RolloutRequest) -> List[OpenAIFunctionToolSchema]:
        return []  # Add your tools here

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)

        for turn in range(ctx.request.max_turns):
            # Log state before LLM call
            ctx.log_event(
                "pre_llm",
                turn=turn,
                num_messages=len(messages),
                messages_summary=[
                    {"role": m["role"], "content_preview": str(m.get("content", ""))[:50]}
                    for m in messages
                ],
            )

            # Call LLM
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

            # Log LLM response
            ctx.log_event(
                "llm_response",
                turn=turn,
                has_tool_calls=result.has_tool_calls,
                finish_reason=result.finish_reason,
                content_preview=str(result.content or "")[:100],
            )

            if not result.has_tool_calls:
                break

            # Process tool calls
            tool_results = []
            for tool_call in result.tool_calls:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                tool_result = await self.execute_tool(name, args)
                tool_results.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                })
                ctx.record_tool_call()

            messages.extend(tool_results)

            # Log tool execution results
            ctx.log_event(
                "tool_results",
                turn=turn,
                num_results=len(tool_results),
                results_summary=[
                    {"tool_call_id": r["tool_call_id"], "content_preview": r["content"][:50]}
                    for r in tool_results
                ],
            )

        # Log completion with reward (if computed)
        reward = self.compute_reward(messages, ctx.request.metadata)
        ctx.log_event(
            "rollout_complete",
            finish_reason="stop" if turn < ctx.request.max_turns - 1 else "max_turns",
            total_turns=turn + 1,
            final_message_count=len(messages),
            reward=reward,
        )

        return ctx.complete(messages, reward=reward)

    async def execute_tool(self, name: str, args: dict) -> str:
        # Your tool execution logic
        return f"Result for {name}"

    def compute_reward(self, messages: list, metadata: dict) -> float:
        # Your reward computation logic
        return 1.0


# Run with: osmosis serve -m logging_agent:agent_loop --log ./rollout_logs
agent_loop = LoggingAgent()
app = create_app(agent_loop)
```

Each server session creates a timestamped subdirectory, and each rollout creates a JSONL file:

```
rollout_logs/
├── 1703270400/           # Unix timestamp when server started
│   ├── rollout-abc123.jsonl
│   └── rollout-def456.jsonl
└── 1703274000/           # Another server session
    └── rollout-xyz789.jsonl
```

Example log file contents:

```jsonl
{"event": "pre_llm", "rollout_id": "rollout-abc123", "turn": 0, "num_messages": 1, ...}
{"event": "llm_response", "rollout_id": "rollout-abc123", "turn": 0, "has_tool_calls": true, ...}
{"event": "tool_results", "rollout_id": "rollout-abc123", "turn": 0, "num_results": 1, ...}
{"event": "pre_llm", "rollout_id": "rollout-abc123", "turn": 1, "num_messages": 4, ...}
{"event": "llm_response", "rollout_id": "rollout-abc123", "turn": 1, "has_tool_calls": false, ...}
{"event": "rollout_complete", "rollout_id": "rollout-abc123", "total_turns": 2, "reward": 1.0, ...}
```

---

## Tool Utilities

The SDK provides utilities for creating and executing tool calls.

```python
# Using tool utilities

from osmosis_ai.rollout import (
    create_tool_result,
    create_tool_error_result,
    serialize_tool_result,
    parse_tool_arguments,
    get_tool_call_info,
    execute_tool_calls,
)

# Create a standardized tool result message
result_msg = create_tool_result("call_123", "42")
# {"role": "tool", "content": "42", "tool_call_id": "call_123"}

# Serialize various types to string
serialize_tool_result(42)           # "42"
serialize_tool_result(3.14)         # "3.14"
serialize_tool_result({"a": 1})     # '{"a": 1}'

# Parse arguments (handles both str and dict formats)
args = parse_tool_arguments('{"a": 5, "b": 3}')
args = parse_tool_arguments({"a": 5, "b": 3})

# Extract tool call info
tool_call = {
    "id": "call_123",
    "type": "function",
    "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'},
}
call_id, name, args = get_tool_call_info(tool_call)
# ("call_123", "add", {"a": 5, "b": 3})

# Execute multiple tool calls concurrently
async def my_executor(tool_call):
    call_id, name, args = get_tool_call_info(tool_call)
    result = await my_tools[name](**args)
    return create_tool_result(call_id, serialize_tool_result(result))

results = await execute_tool_calls(tool_calls, my_executor)
messages.extend(results)

# Handle errors gracefully
try:
    result = await my_tools[name](**args)
    return create_tool_result(call_id, serialize_tool_result(result))
except Exception as e:
    return create_tool_error_result(call_id, str(e))
```

---

## Message Utilities

Helper functions for working with messages.

```python
from osmosis_ai.rollout import (
    parse_tool_calls,
    normalize_stop,
    get_message_content,
    get_message_role,
    is_assistant_message,
    is_tool_message,
    is_user_message,
    count_messages_by_role,
)

# Safely extract tool_calls from assistant message
tool_calls = parse_tool_calls(assistant_message)
if tool_calls:
    for tc in tool_calls:
        # Process tool call...

# Normalize stop parameter (handles None, str, List[str])
stop = normalize_stop(params.get("stop"))  # Always List[str] or None

# Get message content safely
content = get_message_content(message)

# Check message roles
if is_assistant_message(message):
    pass
elif is_tool_message(message):
    pass

# Count messages by role
counts = count_messages_by_role(messages)
# {"user": 3, "assistant": 2, "tool": 1}
```

---

## Using the Registry

For applications with multiple agent types.

```python
from osmosis_ai.rollout import (
    register_agent_loop,
    get_agent_loop,
    list_agent_loops,
    create_app,
)

# Register agents at startup
register_agent_loop(CalculatorAgent())
register_agent_loop(SearchAgent())
register_agent_loop(WeatherAgent())

# List available agents
print(list_agent_loops())  # ['calculator', 'search', 'weather']

# Get specific agent
agent = get_agent_loop("calculator")
app = create_app(agent)
```

## Example Repository

For a complete, runnable project with tools, rewards, and server setup, see: [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example)

## See Also

- [Testing](./testing.md) -- unit tests and mock trainer
- [Deployment](./deployment.md) -- Docker, health checks, production config
- [Agent Loop Guide](./agent-loop.md) -- endpoints, schemas, types
