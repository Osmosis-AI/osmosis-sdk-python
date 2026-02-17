# Osmosis SDK Documentation

Python SDK for Osmosis AI training workflows. The SDK supports two parallel training modes -- **Local Rollout** and **Remote Rollout** -- along with shared tools for testing, evaluation, and reward scoring.

## Getting Started

- [Installation & Quick Start](../README.md)
- [Contributing](../CONTRIBUTING.md)

## Local Rollout

Git-sync mode where Osmosis manages the agent loop. You provide reward functions, rubrics, and MCP tools.

- [Overview](./local-rollout/overview.md) -- what is local rollout, when to choose it, repository structure
- [Reward Functions](./local-rollout/reward-functions.md) -- `@osmosis_reward` decorator usage and examples
- [Reward Rubrics](./local-rollout/reward-rubrics.md) -- `@osmosis_rubric` and `evaluate_rubric` usage
- [MCP Tools](./local-rollout/mcp-tools.md) -- defining `@mcp.tool()` functions for the agent

## Remote Rollout

Self-hosted mode where you implement and run the agent loop as a server.

- [Overview](./remote-rollout/overview.md) -- quick start and key concepts
- [Architecture](./remote-rollout/architecture.md) -- protocol design and agent lifecycle
- [Agent Loop Guide](./remote-rollout/agent-loop.md) -- API reference for classes, schemas, types
- [Examples](./remote-rollout/examples.md) -- agent implementations and utilities
- [Testing](./remote-rollout/testing.md) -- unit tests and mock trainer
- [Deployment](./remote-rollout/deployment.md) -- Docker, health checks, production config

## Shared Concepts

Used by both Local Rollout and Remote Rollout modes.

- [Dataset Format](./datasets.md) -- supported formats and required columns
- [Test Mode](./test-mode.md) -- test agents with cloud LLMs (`osmosis test`)
- [Eval Mode](./eval-mode.md) -- evaluate agents with eval functions and pass@k (`osmosis eval`)

## Reference

- [Rewards API Reference](./rewards-api.md) -- `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric` API details
- [CLI Reference](./cli.md) -- all `osmosis` commands
- [Configuration](./configuration.md) -- environment variables, settings classes, and programmatic configuration
- [Troubleshooting](./troubleshooting.md) -- common errors, debug tips, and resolutions

## Contributing

We welcome contributions! See the [Contributing Guide](../CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Other

- [Example Code](../examples/) -- reward functions, rubric configs, sample data
- [Security Policy](../SECURITY.md)
- [License](../LICENSE)
