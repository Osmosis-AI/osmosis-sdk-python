# {name}

Structured Osmosis project for task-specific training.

## Quick start

```bash
# Install dependencies
pip install -e .

# Authenticate with Osmosis
osmosis auth login

# Verify the canonical project layout
osmosis project validate
```

For AI agents or automation, use `osmosis --json ...` for structured output or
`osmosis --plain ...` for low-noise text.

## Ask your agent

```text
I want to train a model for <task>. Read .osmosis/research/program.md,
create a baseline rollout in this project, iterate locally with evals,
and prepare a training config. Use `osmosis --json` or `osmosis --plain`
for Osmosis CLI commands when you need machine-readable or low-noise output.
```
