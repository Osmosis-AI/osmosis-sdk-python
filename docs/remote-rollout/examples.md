# Examples

Complete working examples for the Osmosis Remote Rollout SDK.

<Note>
Full source code is available in the [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) repository. The snippets below are abbreviated for clarity.
</Note>

## CLI Quick Start

```bash
# Validate agent loop (checks tools, async run, etc.)
osmosis validate -m server:agent_loop

# Start server (default port 9000)
osmosis serve -m server:agent_loop

# Start with auto-reload and debug logging
osmosis serve -m server:agent_loop --reload --log ./logs

# Test locally with a cloud LLM
osmosis test -m server:agent_loop -d test_data.jsonl --model gpt-5-mini

# Evaluate with reward function
osmosis eval -m server:agent_loop -d test_data.jsonl \
    --eval-fn rewards:compute_reward --model gpt-5-mini
```

The module path format is `module:attribute`. The CLI automatically adds the current directory to Python path.

---

## Project Structure

```
rollout-server/
├── server.py        # Agent loop + FastAPI app
├── tools.py         # Tool definitions and execution
├── rewards.py       # Reward computation
├── test_data.jsonl  # Test dataset
└── pyproject.toml   # Dependencies: osmosis-ai[server]>=0.2.14
```

---

## Debug Logging

Enable debug logging with the `--log` CLI flag or `ROLLOUT_DEBUG_DIR` environment variable. Use `ctx.log_event()` in your agent's `run()` method to record events -- these are no-ops unless debug logging is enabled.

```bash
# Via CLI flag
osmosis serve -m server:agent_loop --log ./rollout_logs

# Via environment variable
ROLLOUT_DEBUG_DIR=./rollout_logs osmosis serve -m server:agent_loop
```

Each server session creates a timestamped subdirectory, and each rollout creates a JSONL file with events like `pre_llm`, `llm_response`, `tool_results`, and `rollout_complete`.

---

## Tool Utilities

The `osmosis_ai.rollout.tools` module provides utilities for tool call execution:

| Function | Description |
|----------|-------------|
| `get_tool_call_info(tool_call)` | Extract `(call_id, name, args)` from a tool call dict, parsing JSON arguments |
| `serialize_tool_result(value)` | Convert any value to a string suitable for tool results |
| `create_tool_result(call_id, content)` | Create a `{"role": "tool", ...}` message dict |
| `create_tool_error_result(call_id, error)` | Create an error tool result message |
| `execute_tool_calls(calls, executor)` | Execute multiple tool calls concurrently via an async executor function |

See the example repo's `tools.py` for the recommended pattern using these utilities.

---

## Message Utilities

The `osmosis_ai.rollout` module exports helpers for working with messages:

| Function | Description |
|----------|-------------|
| `parse_tool_calls(message)` | Safely extract `tool_calls` from an assistant message |
| `normalize_stop(value)` | Normalize stop parameter to `List[str]` or `None` |
| `get_message_content(message)` | Get message content safely |
| `is_assistant_message(message)` | Check if message role is `assistant` |
| `is_tool_message(message)` | Check if message role is `tool` |
| `is_user_message(message)` | Check if message role is `user` |
| `count_messages_by_role(messages)` | Count messages by role, returns `{"user": N, ...}` |

---

## See Also

- [Testing](./testing.md) -- unit tests and mock trainer
- [Agent Loop Guide](./agent-loop.md) -- endpoints, schemas, types
