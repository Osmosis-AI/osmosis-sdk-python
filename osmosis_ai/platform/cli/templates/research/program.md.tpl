# Local Research Loop

Use this file to steer short local iteration loops before launching a long
platform training run.

## Goal

Describe the current task, success criteria, and what behavior you want the
rollout to improve.

## Constraints

- Keep project changes inside canonical paths.
- Prefer small diffs and quick local evals.
- Avoid expensive platform training until the local baseline looks healthy.

## Iteration loop

1. Read the current rollout, grader, dataset, and config files.
2. Pick one hypothesis for improvement.
3. Make the smallest change that tests that hypothesis.
4. Run `osmosis --json eval run configs/eval/<name>.toml`.
5. Compare results to the previous baseline.
6. Keep or revert the change based on the result.
7. Log the experiment under `.osmosis/research/experiments/`.
