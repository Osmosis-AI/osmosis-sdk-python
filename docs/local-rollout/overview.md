# Local Rollout

Local Rollout is one of two training modes supported by the Osmosis platform. In this mode, Osmosis manages the entire agent loop -- you provide **reward functions**, **rubric evaluators**, and optionally **MCP tools** via a git-sync repository.

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
│   └── tools/                  # Tool implementations
├── reward_fn/                  # Reward functions (deterministic scoring)
│   └── compute_reward.py
├── reward_rubric/              # Rubric evaluators (LLM-as-judge scoring)
│   └── reward_rubric_openai.py
├── test_data.jsonl             # Sample dataset for local testing
└── pyproject.toml
```

All three directories are optional -- include only what your training run needs.

## Local Testing

```bash
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

## Example Repository

See the complete working example: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)

## Next Steps

- [Reward Functions](./reward-functions.md) -- define `@osmosis_reward` scoring functions
- [Reward Rubrics](./reward-rubrics.md) -- define `@osmosis_rubric` LLM-as-judge evaluators
- [MCP Tools](./mcp-tools.md) -- define `@mcp.tool()` functions for the agent
- [Dataset Format](../datasets.md) -- dataset format for testing and evaluation
