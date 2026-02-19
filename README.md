<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/osmosis-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset=".github/osmosis-logo-light.svg">
    <img alt="Osmosis" src=".github/osmosis-logo-light.svg" width="218">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue"></a>
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/osmosis-ai?color=yellow"></a>
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/osmosis-ai"></a>
  <a href="https://codecov.io/gh/Osmosis-AI/osmosis-sdk-python"><img alt="Codecov" src="https://codecov.io/gh/Osmosis-AI/osmosis-sdk-python/branch/main/graph/badge.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-orange.svg"></a>
  <a href="https://docs.osmosis.ai"><img alt="Docs" src="https://img.shields.io/badge/docs-docs.osmosis.ai-green"></a>
</p>

# osmosis-ai

> ⚠️ **Warning**: osmosis-ai is still in active development. APIs may change between versions.

Python SDK for [Osmosis AI](https://platform.osmosis.ai), a platform for training LLMs with reinforcement learning. Define custom reward functions, LLM-as-judge rubrics, and agent tools -- then use this SDK to build and test them locally before submitting training runs on managed GPU clusters.

## Quick Start

Pick a training mode and follow the example repo:

| | Local Rollout | Remote Rollout |
|--|--------------|----------------|
| **How it works** | Osmosis manages the agent loop. You provide reward functions, rubrics, and MCP tools via a GitHub-synced repo. | You implement and host a `RolloutAgentLoop` server. Full control over agent behavior. |
| **Best for** | Standard tool-use agents, fast iteration, zero infrastructure | Custom agent architectures, complex orchestration, persistent environments |
| **Example repo** | [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) | [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) |
| **Docs** | [Local Rollout Guide](docs/local-rollout/overview.md) | [Remote Rollout Guide](docs/remote-rollout/overview.md) |

## Installation

Requires **Python 3.10+**. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

- **An LLM API key** (e.g., OpenAI, Anthropic, Groq) -- required for `osmosis test` and `osmosis eval`. See [supported providers](https://docs.litellm.ai/docs/providers).
- **Osmosis account** (optional) -- needed for platform features like `osmosis login`, workspace management, and submitting training runs. Sign up at [platform.osmosis.ai](https://platform.osmosis.ai).

**pip**

```bash
pip install osmosis-ai            # Core SDK
pip install osmosis-ai[server]    # + FastAPI server
pip install osmosis-ai[mcp]       # + MCP tool support
pip install osmosis-ai[full]      # All features
```

**uv**

```bash
uv add osmosis-ai                 # Core SDK
uv add osmosis-ai[server]         # + FastAPI server
uv add osmosis-ai[mcp]            # + MCP tool support
uv add osmosis-ai[full]           # All features
```

## Testing & Evaluation

Both modes share the same CLI tools: [Test Mode](docs/test-mode.md) | [Eval Mode](docs/eval-mode.md) | [CLI Reference](docs/cli.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
