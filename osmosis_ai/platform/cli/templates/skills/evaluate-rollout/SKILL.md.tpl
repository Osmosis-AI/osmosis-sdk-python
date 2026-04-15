# Skill: Evaluate Rollout

## Goal
Help the user test and evaluate a rollout before submitting training runs.

## Steps

1. Identify which rollout to evaluate (check `rollouts/` directory)
2. Ensure `configs/eval/<env_name>.toml` points at the rollout entrypoint and dataset
3. Run evaluation: `osmosis eval run configs/eval/<env_name>.toml`
4. Analyze reward distribution and identify issues
5. Iterate on tools/grader if needed

## Common Issues

- Tools returning non-string values (model can't read them)
- Grader too strict/lenient (check reward distribution)
- System prompt unclear (model doesn't use tools correctly)
- max_turns too low (agent can't complete complex tasks)
