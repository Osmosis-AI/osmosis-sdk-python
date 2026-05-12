# {name}

Structured Osmosis project for task-specific training.

This repository was created from an Osmosis Platform project scaffold. Clone the
repository created by the Platform, then run CLI commands from that checkout.

## Quick start

```bash
# Clone the repository created by the Platform
git clone <repo-url>
cd <repo>

# Install dependencies
pip install -e .

# Authenticate with Osmosis
osmosis auth login

# Verify the canonical project layout
osmosis project validate

# Edit rollouts under rollouts/
$EDITOR rollouts/

# Submit the default training config
osmosis train submit configs/training/default.toml
```

Platform-scoped commands derive scope from this checkout's `origin` remote.
`.osmosis/` contains local runtime state and is ignored by Git.

For AI agents or automation, use `osmosis --json ...` for structured output or
`osmosis --plain ...` for low-noise text.

## Ask your agent

```text
I want to train a model for <task>. Read research/program.md,
create a baseline rollout in this project, iterate locally with evals,
and prepare a training config. Use `osmosis --json` or `osmosis --plain`
for Osmosis CLI commands when you need machine-readable or low-noise output.
```
