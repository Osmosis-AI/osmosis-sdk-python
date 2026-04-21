# Osmosis Workspace

## Overview

This is an Osmosis workspace. It contains training rollouts for reinforcement learning, synced to the Osmosis Platform via Git.

## Structure

- `rollouts/<env_name>/` — Each rollout is self-contained
  - `main.py` — **Entry point** (required). Defines a concrete `AgentWorkflow` and typically a concrete `Grader`.
  - `pyproject.toml` — Per-rollout dependencies.
  - `README.md` — Rollout description.
- `configs/training/` — Training configurations (TOML, workspace-scoped). Each config references a rollout by name.
- `configs/eval/` — Evaluation configurations (TOML).
- `data/` — Local test data (JSONL).
- `.osmosis/workspace.toml` — Workspace metadata. Do not edit manually.

## Conventions

### Rollout Definition

Each rollout lives in `rollouts/<env_name>/` and should expose one concrete `AgentWorkflow` subclass in the entrypoint module. For `osmosis rollout serve` and most eval flows, that module should also expose a concrete `Grader` and usually a `GraderConfig`.

### Tools

Define tools as plain Python async functions with type hints and docstrings.
The SDK automatically generates OpenAI-compatible tool schemas from the function signature.

### Grader

Define grader functions that score rollout results. Return a float between 0.0 and 1.0.
Always include `**kwargs` for forward compatibility.

## CLI Commands

```bash
# Local development
osmosis eval run configs/eval/<env_name>.toml   # Batch evaluation
osmosis rollout serve <path-to-serve-config.toml>  # Start rollout server

# Training (workspace-scoped)
osmosis train submit configs/training/qwen3-4b.toml
osmosis train status <run-name>
```
