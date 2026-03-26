# Osmosis Workspace

## Overview

This is an Osmosis workspace. It contains training environments for reinforcement learning, synced to the Osmosis Platform via Git.

## Structure

- `environments/<env_name>/` — Each environment is self-contained
  - `main.py` — **Entry point** (required). Must export `load_environment()`.
  - `pyproject.toml` — Per-environment dependencies.
  - `README.md` — Environment description.
- `configs/training/` — Training configurations (TOML, workspace-scoped). Each config references an environment by name.
- `configs/eval/` — Evaluation configurations (TOML).
- `data/` — Local test data (JSONL).
- `.osmosis/workspace.toml` — Workspace metadata. Do not edit manually.

## Conventions

### Environment Definition

Each environment lives in `environments/<env_name>/` and must have a `main.py` with a `load_environment()` function.

### Tools

Define tools as plain Python async functions with type hints and docstrings.
The SDK automatically generates OpenAI-compatible tool schemas from the function signature.

### Grader

Define grader functions that score rollout results. Return a float between 0.0 and 1.0.
Always include `**kwargs` for forward compatibility.

## CLI Commands

```bash
# Local development
osmosis test <env_name> -m gpt-4.1-mini        # Quick test with cloud LLM
osmosis eval <env_name> -d data/test.jsonl      # Batch evaluation
osmosis serve <env_name>                        # Start rollout server

# Training (workspace-scoped)
osmosis train submit configs/training/qwen3-4b.toml
osmosis train status <run-id>
```
