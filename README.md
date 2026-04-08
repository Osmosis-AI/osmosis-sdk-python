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

Python SDK for [Osmosis AI](https://platform.osmosis.ai), a platform for training LLMs with reinforcement learning. Implement an **AgentWorkflow** in Python, add a concrete **Grader** for serve/eval flows, drive them locally with the CLI, then connect a **RolloutServer** to managed training.

## Quick start

| Step | What you do |
|------|-------------|
| **Define agents** | One `AgentWorkflow` subclass (+ optional `AgentWorkflowConfig`) in your repo. For `rollout serve`, the entrypoint must also expose a concrete `Grader` (typically with a `GraderConfig`). |
| **Layout** | Use a rollout pack directory under `rollouts/<name>/` when loading by rollout name; the CLI adds that directory to `sys.path`. |
| **Serve** | `osmosis rollout serve serve.toml` — HTTP server for TrainGate. The entrypoint module must expose a concrete `Grader` (typically with a `GraderConfig`); config is TOML (`[serve]`, `[server]`, `[registration]`, `[debug]`). |
| **Evaluate** | `osmosis eval run eval.toml` — same execution stack as training, with optional pass@k and caching. |

**Example repositories:** [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) (synced agent repo patterns) · [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) (reference server usage — align with current SDK exports when upgrading).

**Documentation index:** [docs/README.md](docs/README.md)

## Installation

Requires **Python 3.12+**. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

- **An LLM API key** (e.g., OpenAI, Anthropic, Groq) — required for `osmosis eval run` when using hosted models. See [supported providers](https://docs.litellm.ai/docs/providers).
- **Osmosis account** (optional) — needed for `osmosis auth login`, workspace management, and registering a rollout server with the platform. Sign up at [platform.osmosis.ai](https://platform.osmosis.ai).

**pip**

```bash
pip install osmosis-ai            # Core SDK
pip install osmosis-ai[server]    # + FastAPI rollout server (pulls in platform extra)
pip install osmosis-ai[full]      # Same as [server] (all packaged optional features)
```

**uv**

```bash
uv add osmosis-ai                 # Core SDK
uv add osmosis-ai[server]         # + FastAPI rollout server (pulls in platform extra)
uv add osmosis-ai[full]           # Same as [server] (all packaged optional features)
```

## Testing and evaluation

- [Eval mode](docs/eval-mode.md) — graded runs, pass@k, cache/resume with `osmosis eval run`
- [CLI reference](docs/cli.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
