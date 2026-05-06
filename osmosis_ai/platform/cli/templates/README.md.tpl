# {name}

Structured Osmosis project for task-specific training.

## Quick start

```bash
# Install dependencies
pip install -e .

# Authenticate with Osmosis
osmosis auth login

# Verify the canonical project layout
osmosis --json project validate
```

## Ask your agent

```text
I want to train a model for <task>. Read .osmosis/research/program.md,
create a baseline rollout in this project, iterate locally with evals,
and prepare a training config. Use `osmosis --json` for Osmosis CLI commands.
```
