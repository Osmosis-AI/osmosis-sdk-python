# Osmosis Remote Rollout SDK

A lightweight SDK for integrating agent frameworks with the Osmosis remote rollout protocol. This SDK handles protocol communication between your agent logic and the Osmosis training infrastructure.

For an alternative approach that requires no server infrastructure, see [Local Rollout](../local-rollout/overview.md).

## Quick Start

```python
class CalculatorAgentLoop(RolloutAgentLoop):
    name = "calculator"

    def get_tools(self, request: RolloutRequest):
        return CALCULATOR_TOOL_SCHEMAS

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)
        for _turn in range(ctx.request.max_turns):
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)
            if not result.has_tool_calls:
                break
            tool_results = await execute_tools(result.tool_calls)
            messages.extend(tool_results)
        return ctx.complete(messages, reward=compute_reward(...))

agent_loop = CalculatorAgentLoop()
app = create_app(agent_loop)
```

```bash
# Validate agent loop
osmosis validate -m server:agent_loop

# Start server (default port 9000)
osmosis serve -m server:agent_loop

# Test locally with a cloud LLM
osmosis test -m server:agent_loop -d test_data.jsonl --model gpt-5-mini
```

For the complete working project, see [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example).

## Next Steps

- [Architecture](./architecture.md) -- Protocol design and flow diagrams
- [Agent Loop Guide](./agent-loop.md) -- Complete API documentation
- [Examples](./examples.md) -- CLI usage and debug logging
