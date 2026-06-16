<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Osmosis-AI/osmosis-sdk-python/main/.github/osmosis-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Osmosis-AI/osmosis-sdk-python/main/.github/osmosis-logo-light.svg">
    <img alt="Osmosis" src="https://raw.githubusercontent.com/Osmosis-AI/osmosis-sdk-python/main/.github/osmosis-logo-light.svg" width="218">
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

Python SDK and CLI for [Osmosis AI](https://platform.osmosis.ai), a platform for training LLMs with reinforcement learning. Implement an **AgentWorkflow** and a concrete **Grader** in Python, then use the CLI to submit evaluation and training runs from an Osmosis workspace directory.

## Installation

Requires **Python 3.12+**.

```bash
pip install osmosis-ai            # Core SDK
pip install osmosis-ai[server]    # + FastAPI rollout server
# or with uv:  uv add osmosis-ai
```

See [Installation](https://docs.osmosis.ai/cli/installation) for the full extras matrix and [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Documentation

Guides, quickstart, and the full CLI reference live at **[docs.osmosis.ai](https://docs.osmosis.ai)**.

- [Quickstart](https://docs.osmosis.ai/platform/quickstart) — run the multiply example end to end, from onboarding to evaluation run to training run
- [CLI command reference](https://docs.osmosis.ai/cli/command-reference) — every `osmosis` command and flag, plus the `--json` / `--plain` output contract for AI agents and CI/CD
- [Workspace setup](https://docs.osmosis.ai/cli/workspace/overview) — repository layout, config files, and Git Sync
- [Rollouts](https://docs.osmosis.ai/cli/rollout/overview) — AgentWorkflow, Grader, integrations, and execution backends
- [Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases) — version history and breaking changes between releases

Building on or contributing to the SDK itself? See the code-anchored developer docs in [`docs/`](docs/) (start with [`docs/architecture.md`](docs/architecture.md)) alongside [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
