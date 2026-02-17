# osmosis-ai

Python SDK for Osmosis AI training workflows. Supports two training modes with shared tooling for testing and evaluation.

## Two Training Modes

Osmosis supports **Local Rollout** and **Remote Rollout** as parallel approaches to training with reinforcement learning:

| | Local Rollout | Remote Rollout |
|--|--------------|----------------|
| **How it works** | Osmosis manages the agent loop. You provide reward functions, rubrics, and MCP tools via a GitHub-synced repo. | You implement and host a `RolloutAgentLoop` server. Full control over agent behavior. |
| **Best for** | Standard tool-use agents, fast iteration, zero infrastructure | Custom agent architectures, complex orchestration, persistent environments |
| **Example repo** | [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) | [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) |

## Installation

Requires Python 3.10 or newer. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

### pip

```bash
pip install osmosis-ai            # Core SDK
pip install osmosis-ai[server]    # FastAPI server for Remote Rollout
pip install osmosis-ai[mcp]       # MCP tool support for Local Rollout
pip install osmosis-ai[full]      # All features
```

### uv

```bash
uv add osmosis-ai                 # Core SDK
uv add osmosis-ai[server]         # FastAPI server for Remote Rollout
uv add osmosis-ai[mcp]            # MCP tool support for Local Rollout
uv add osmosis-ai[full]           # All features
```

## Local Rollout

Osmosis manages the agent loop. You provide reward functions, rubrics, and MCP tools via a GitHub-synced repo. Best for standard tool-use agents, fast iteration, and zero infrastructure.

Get started with the example repo: **[osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)**

For details, see the [Local Rollout docs](docs/local-rollout/overview.md).

## Remote Rollout

You implement and host a `RolloutAgentLoop` server. Full control over agent behavior, custom orchestration, and persistent environments.

Get started with the example repo: **[osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example)**

For details, see the [Remote Rollout docs](docs/remote-rollout/overview.md).

## Testing & Evaluation

Both modes share the same `osmosis test` and `osmosis eval` CLI tools. See the example repos for usage:

- **Local Rollout**: [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example)
- **Remote Rollout**: [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example)

For CLI details, see [Test Mode](docs/test-mode.md), [Eval Mode](docs/eval-mode.md), and the [CLI Reference](docs/cli.md).

## Documentation

| Section | Topics |
|---------|--------|
| **Local Rollout** | |
| [Overview](docs/local-rollout/overview.md) | When to choose, repo structure, setup |
| [Reward Functions](docs/local-rollout/reward-functions.md) | `@osmosis_reward` decorator |
| [Reward Rubrics](docs/local-rollout/reward-rubrics.md) | `@osmosis_rubric`, `evaluate_rubric` |
| [MCP Tools](docs/local-rollout/mcp-tools.md) | `@mcp.tool()` definitions |
| **Remote Rollout** | |
| [Overview](docs/remote-rollout/overview.md) | Quick start, key concepts |
| [Architecture](docs/remote-rollout/architecture.md) | Protocol design and lifecycle |
| [Agent Loop Guide](docs/remote-rollout/agent-loop.md) | API reference |
| [Examples](docs/remote-rollout/examples.md) | Agent implementations |
| [Testing](docs/remote-rollout/testing.md) | Unit tests and mock trainer |
| [Deployment](docs/remote-rollout/deployment.md) | Docker, production config |
| **Shared** | |
| [Dataset Format](docs/datasets.md) | Parquet, JSONL, CSV formats |
| [Test Mode](docs/test-mode.md) | `osmosis test` |
| [Eval Mode](docs/eval-mode.md) | `osmosis eval` with pass@k |
| **Reference** | |
| [Rewards API](docs/rewards-api.md) | Decorator and function signatures |
| [CLI Reference](docs/cli.md) | All `osmosis` commands |

Full documentation index: [docs/README.md](docs/README.md)

## Examples

The [`examples/`](examples/) directory contains standalone SDK API examples (reward functions, rubric evaluation). See [`examples/README.md`](examples/README.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [Homepage](https://github.com/Osmosis-AI/osmosis-sdk-python)
- [Issues](https://github.com/Osmosis-AI/osmosis-sdk-python/issues)
- [Local Rollout Example](https://github.com/Osmosis-AI/osmosis-git-sync-example)
- [Remote Rollout Example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example)
