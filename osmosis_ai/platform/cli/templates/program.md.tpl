# Training Brief

## Task Goal

Describe the behavior this project is training or evaluating.

## Success Criteria

- Define the reward, rubric, or measurable behavior that counts as success.
- Include the minimum baseline quality needed before platform training.

## Dataset Constraints

- Document required columns, size assumptions, and examples that should be excluded.

## Known Risks

- List failure modes the rollout or grader should catch.

## Training Plan

1. Build a baseline rollout under `rollouts/`.
2. Run local evals with `osmosis eval run configs/eval/<name>.toml`.
3. Submit training with `osmosis train submit configs/training/<name>.toml --yes` after the linked workspace Git Sync repository is configured and pushed.
