# Skill: Submit Training Run

## Goal
Help the user configure and submit a training run on the Osmosis Platform.

## Steps

1. Identify which environment to train (check `environments/` directory)
2. Ensure environment is validated (`osmosis test <env_name>` passes)
3. Choose or create a training config in `configs/training/`
4. Ensure the config has `environment = "<env_name>"` at the top level
5. Verify dataset is available (workspace-scoped or local file)
6. Submit: `osmosis train submit configs/training/<config>.toml`
7. Monitor: `osmosis train status <run-id>`

## Training Config Checklist

- [ ] `environment` field references a valid environment name
- [ ] Model ID is valid HuggingFace model path
- [ ] Dataset exists and is accessible
- [ ] Learning rate is reasonable (1e-5 to 1e-4 typical)
- [ ] batch_size * rollout_n fits in GPU memory
- [ ] max_steps is appropriate for dataset size
- [ ] LoRA rank set (0 for full fine-tune, 16-64 for LoRA)

## Execution Modes

- **Platform-managed**: Just push code, platform hosts the environment
- **Remote rollout**: User runs `osmosis serve <env_name>`, platform connects to it
