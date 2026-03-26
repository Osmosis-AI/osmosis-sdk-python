# Training & Evaluation Configs

Configs are workspace-scoped and reference specific environments by name.

## Training Configs (`training/*.toml`)

Each TOML file defines a training run configuration. You can have multiple configs
for different model/hyperparameter/environment combinations.

```toml
environment = "calculator"            # Required: target environment name
model = "Qwen/Qwen3-4B"              # HuggingFace model ID

[training]
batch_size = 128
learning_rate = 0.00001
max_steps = 100
rollout_n = 8                         # Rollouts per example
lora_rank = 16                        # 0 = full fine-tune
lora_alpha = 32
max_prompt_length = 8192
max_response_length = 8192

[sampling]
max_tokens = 512
temperature = 0.7

[data]
dataset = "workspace-dataset-name"    # Workspace-scoped dataset
# dataset_file = "data/train.jsonl"  # Or local file (auto-uploaded)
```

## Eval Configs (`eval/*.toml`)

```toml
environment = "calculator"            # Required: target environment name

[eval]
num_examples = 20
rollouts_per_example = 3

[sampling]
max_tokens = 512
temperature = 0.7
```

## Commands

```bash
osmosis train submit configs/training/qwen3-4b.toml
osmosis eval calculator -c configs/eval/default.toml -m gpt-4.1-mini
```
