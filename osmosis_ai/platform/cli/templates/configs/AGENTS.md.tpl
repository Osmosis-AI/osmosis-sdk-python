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
entrypoint = "main.py"
model_path = "Qwen/Qwen3.5-35B-A3B"  # or "Qwen/Qwen3.5-122B-A10B"
dataset = "my-dataset-abc123"

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
[eval]
rollout = "calculator"
entrypoint = "main.py"
dataset = "data/dataset.jsonl"

[llm]
model = "openai/gpt-5-mini"

[runs]
n = 3
batch_size = 2
```

## Commands

```bash
osmosis train submit configs/training/<config>.toml
osmosis eval run configs/eval/<config>.toml
osmosis rollout serve <path-to-serve-config.toml>
osmosis train status <run-name>
```
