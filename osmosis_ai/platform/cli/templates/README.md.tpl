# {name}

Structured Osmosis workspace for task-specific training.

## Quick start

```bash
# Install dependencies
pip install -e .

# Authenticate with Osmosis
osmosis auth login

# Verify the canonical workspace layout
osmosis --json workspace validate
```

## Ask your agent

```text
I want to train a model for <task>. Read .osmosis/research/program.md,
create a baseline rollout in this workspace, iterate locally with evals,
and prepare a training config. Use `osmosis --json` for Osmosis CLI commands.
```
