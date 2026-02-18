# Examples

Full source code is available in the [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) repository.

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

## Debug Logging

Enable debug logging with the `--log` CLI flag or `ROLLOUT_DEBUG_DIR` environment variable. Use `ctx.log_event()` in your agent's `run()` method to record events -- these are no-ops unless debug logging is enabled.

```bash
# Via CLI flag
osmosis serve -m server:agent_loop --log ./rollout_logs

# Via environment variable
ROLLOUT_DEBUG_DIR=./rollout_logs osmosis serve -m server:agent_loop
```

Each server session creates a timestamped subdirectory, and each rollout creates a JSONL file with events like `pre_llm`, `llm_response`, `tool_results`, and `rollout_complete`.

## See Also

- [Agent Loop Guide](./agent-loop.md) -- endpoints, schemas, types
