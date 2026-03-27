# Training & Evaluation Configs

Configs are workspace-scoped and reference specific rollouts by name.

## Training Configs (`training/*.toml`)

Each TOML file defines a training run configuration. Start from the default template:

```bash
cp configs/training/default.toml configs/training/<run_name>.toml
```

Then fill in required fields and adjust parameters as needed. See the template for all available options with defaults.

### Example

```toml
[experiment]
rollout = "calculator"
model_path = "Qwen/Qwen3.5-35B-A3B"  # or "Qwen/Qwen3.5-122B-A10B"
dataset_id = "my-dataset-abc123"

[training]
# lr = 1e-6
# total_epochs = 1
# n_samples_per_prompt = 8
# global_batch_size = 64
# max_prompt_length = 8192
# max_response_length = 8192

[sampling]
# rollout_temperature = 1.0
# rollout_top_p = 1.0

[checkpoints]
# eval_interval =
# checkpoint_save_freq = 20
```

## Eval Configs (`eval/*.toml`)

```toml
rollout = "calculator"            # Required: target rollout name

[eval]
num_examples = 20
rollouts_per_example = 3

[sampling]
max_tokens = 512
temperature = 0.7
```

## Commands

```bash
osmosis train submit configs/training/<config>.toml
osmosis eval <env_name> -c configs/eval/default.toml -m gpt-4.1-mini
```
