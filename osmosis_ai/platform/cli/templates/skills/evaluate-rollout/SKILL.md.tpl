# Skill: Evaluate Rollout

## Goal
Help the user test and evaluate a rollout before submitting training runs.

## Steps

1. Identify which rollout to evaluate (check `rollouts/` directory)
2. Check that `rollouts/<env_name>/main.py` loads without errors
3. Run quick smoke test: `osmosis test <env_name> -m gpt-4.1-mini`
4. Review test output for tool execution and grading
5. Run full evaluation: `osmosis eval <env_name> -d data/test_samples.jsonl -m gpt-4.1-mini`
6. Analyze reward distribution and identify issues
7. Iterate on tools/grader if needed

## Common Issues

- Tools returning non-string values (model can't read them)
- Grader too strict/lenient (check reward distribution)
- System prompt unclear (model doesn't use tools correctly)
- max_turns too low (agent can't complete complex tasks)
