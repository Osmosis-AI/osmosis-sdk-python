# Rollout server configuration.
# Command: uv run osmosis rollout serve rollout.toml

backend = "simple"

[simple]
workflow = "main:MyAgentWorkflow"
grader = "main:MyGrader"
