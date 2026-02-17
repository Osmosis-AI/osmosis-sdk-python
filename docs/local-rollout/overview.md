# Local Rollout

Local Rollout is one of two training modes supported by the Osmosis platform. In this mode, Osmosis manages the entire agent loop -- you provide **reward functions**, **rubric evaluators**, and optionally **MCP tools** via a GitHub-synced repository. The training infrastructure handles LLM inference, tool execution, and trajectory collection automatically.

## When to Choose Local Rollout

| Consideration | Local Rollout | Remote Rollout |
|---------------|--------------|----------------|
| **Agent logic** | Managed by Osmosis | You implement `RolloutAgentLoop` |
| **Tool execution** | MCP tools synced from GitHub | Custom tool code in your server |
| **Infrastructure** | None -- Osmosis handles everything | You host a RolloutServer |
| **Flexibility** | Standard agent loop, configurable tools | Full control over agent behavior |
| **Best for** | Tool-use tasks with standard agent patterns | Custom agent architectures, complex orchestration |

Choose **Local Rollout** when:
- Your task follows a standard tool-use agent pattern (LLM calls tools in a loop)
- You want zero infrastructure to manage
- Your tools can be expressed as MCP `@mcp.tool()` functions
- You want fast iteration via git push

Choose **Remote Rollout** when you need full control over the agent loop, custom orchestration logic, or tools that require a persistent server environment. See the [Remote Rollout overview](../remote-rollout/overview.md).

## How It Works

1. **Connect your GitHub repo** to Osmosis via the platform UI
2. Osmosis syncs your repo and discovers:
   - `reward_fn/` -- `@osmosis_reward` functions for deterministic scoring
   - `reward_rubric/` -- `@osmosis_rubric` functions for LLM-as-judge scoring
   - `mcp/` -- `@mcp.tool()` functions the agent can call during rollouts
3. **Create a training run** on the platform, selecting "Local Rollout" mode
4. Osmosis runs the agent loop on its training cluster, calling your MCP tools and scoring outputs with your reward/rubric functions

## Required Repository Structure

```
your-repo/
├── mcp/                        # MCP tools (agent's callable functions)
│   ├── main.py                 # Entry point -- creates FastMCP instance
│   ├── server/                 # Server setup
│   │   └── mcp_server.py
│   └── tools/                  # Tool implementations
│       ├── __init__.py
│       └── math.py
├── reward_fn/                  # Reward functions (deterministic scoring)
│   └── compute_reward.py
├── reward_rubric/              # Rubric evaluators (LLM-as-judge scoring)
│   ├── reward_rubric_openai.py
│   └── reward_rubric_anthropic.py
├── test_data.jsonl             # Sample dataset for local testing
└── pyproject.toml
```

All three directories (`mcp/`, `reward_fn/`, `reward_rubric/`) are optional -- include only what your training run needs.

## Local Testing

Before pushing to GitHub, test your setup locally:

```bash
# Install MCP support
pip install "osmosis-ai[mcp]"

# Test MCP tools against a dataset
osmosis test --mcp ./mcp -d test_data.jsonl --model openai/gpt-5-mini

# Interactive debugging
osmosis test --mcp ./mcp -d test_data.jsonl --interactive

# Evaluate with your reward function
osmosis eval --mcp ./mcp -d test_data.jsonl \
    --eval-fn reward_fn.compute_reward:numbers_match_reward \
    --model openai/gpt-5-mini
```

See [Test Mode](../test-mode.md) and [Eval Mode](../eval-mode.md) for full documentation.

## Example Repository

See the complete working example: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)

## Next Steps

- [Reward Functions](./reward-functions.md) -- define `@osmosis_reward` scoring functions
- [Reward Rubrics](./reward-rubrics.md) -- define `@osmosis_rubric` LLM-as-judge evaluators
- [MCP Tools](./mcp-tools.md) -- define `@mcp.tool()` functions for the agent
- [Dataset Format](../datasets.md) -- dataset format for testing and evaluation
