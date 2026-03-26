# Skill: Submit Training Run

## Goal
Help the user configure and submit a training run on the Osmosis Platform.

## Steps

1. Identify which environment to train (check `environments/` directory)
2. Ensure environment is validated (`osmosis test <env_name>` passes)
3. Copy the default template: `cp configs/training/default.toml configs/training/<run_name>.toml`
4. Fill in the required `[experiment]` fields (`environment`, `model_path`, `dataset_id`)
5. Adjust optional hyperparameters based on the scenario (see guidance below)
6. Verify dataset is available (`osmosis dataset list`)
7. Submit: `osmosis train submit configs/training/<run_name>.toml`
8. Monitor: `osmosis train status <run-id>`

## Config Template

Start from `configs/training/default.toml`. It contains all available parameters with defaults documented inline.

### `[experiment]` — Required

| Field | What to fill in |
|-------|-----------------|
| `environment` | Environment name — must match a directory under `environments/` |
| `model_path` | Must be one of: `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-122B-A10B` |
| `dataset_id` | Dataset ID from `osmosis dataset list` |
| `commit_sha` | *(optional)* Git commit SHA to pin the environment code. If omitted, uses latest on default branch. |

### Scenario-Based Tuning Guide

Parameters below are in `[training]` unless noted otherwise.

**Quick experiment** (validate setup works):
- `total_epochs = 1`, `global_batch_size = 64`
- Keep all other defaults

**Qwen3.5-35B-A3B** (lighter, faster iterations):
- `lr = 1e-5` to `5e-5` (can be more aggressive)
- `n_samples_per_prompt = 4` to `8`
- Lower `max_prompt_length` / `max_response_length` if your task has short inputs

**Qwen3.5-122B-A10B** (larger, more capable):
- `lr = 1e-6` to `5e-6` (conservative)
- `n_samples_per_prompt = 8` to `16`
- `checkpoint_save_freq = 10` in `[checkpoints]` (save more often, longer runs)

**Long-context tasks**:
- Increase `max_prompt_length` / `max_response_length` as needed (up to model limit)
- May need smaller `global_batch_size` to fit in memory

### Parameter Constraints

- `[training]` `global_batch_size` must be divisible by `n_samples_per_prompt`
- Higher `n_samples_per_prompt` improves reward signal but uses more GPU memory
- `[sampling]` `rollout_temperature > 1.0` increases exploration but may produce incoherent outputs

## Execution Modes

- **Platform-managed**: Push code via Git Sync, platform hosts the environment
- **Remote rollout**: User runs `osmosis serve <env_name>`, platform connects to it
