# Osmosis SDK Documentation

Python SDK for Osmosis AI training workflows, providing reward function validation, rubric evaluation via LLM-as-judge, a Remote Rollout SDK for integrating agent frameworks with Osmosis training, and CLI tools for testing, evaluation, and workspace management.

## Getting Started

- [Installation & Quick Start](../README.md)
- [Contributing](../CONTRIBUTING.md)

## Rewards & Rubrics

- [Rewards & Rubrics Guide](./rewards.md) -- `@osmosis_reward`, `@osmosis_rubric`, `evaluate_rubric`
- [Example Code](../examples/) -- reward functions, rubric configs, sample data

## Remote Rollout SDK

- [Quick Start](./rollout/README.md) -- get a rollout server running in minutes
- [Architecture](./rollout/architecture.md) -- protocol design and agent lifecycle
- [API Reference](./rollout/api-reference.md) -- endpoints, schemas, types
- [Examples](./rollout/examples.md) -- agent implementations and utilities
- [Dataset Format](./rollout/dataset-format.md) -- supported formats and required columns
- [Test Mode](./rollout/test-mode.md) -- test agents with cloud LLMs
- [Evaluation](./rollout/eval.md) -- evaluate agents against datasets
- [Testing](./rollout/testing.md) -- unit tests and mock trainer
- [Deployment](./rollout/deployment.md) -- Docker, health checks, production config

## CLI Reference

- [CLI Reference](./cli.md) -- all `osmosis` commands

## Other

- [Security Policy](../SECURITY.md)
- [License](../LICENSE)
