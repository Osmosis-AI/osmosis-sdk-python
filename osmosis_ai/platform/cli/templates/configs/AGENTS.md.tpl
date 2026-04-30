# Training & Evaluation Configs

Configs are project-scoped and must stay in their canonical directories.

## Canonical paths

- Training: `configs/training/<name>.toml`
- Eval: `configs/eval/<name>.toml`

Do not place these configs elsewhere. The CLI validates these locations.
When running `osmosis` commands as an AI agent or in automation, use the global
`--json` flag before the command.

## Training configs (`training/*.toml`)

Start from the default template:

```bash
cp configs/training/default.toml configs/training/<run_name>.toml
```

Then fill in the required fields:

- `rollout` must match a directory under `rollouts/`
- `entrypoint` must be a Python path relative to that rollout, usually `main.py`
- `model_path` must be a supported base model
- `dataset` must be a platform dataset name

## Eval configs (`eval/*.toml`)

Use one eval config per rollout baseline. `entrypoint` should usually be `main.py`.

```toml
[eval]
rollout = "calculator"
entrypoint = "main.py"
dataset = "data/calculator.jsonl"

[llm]
model = "openai/gpt-5-mini"

[runs]
n = 3
batch_size = 2
```

## Commands

```bash
osmosis --json project validate
osmosis --json rollout validate configs/training/<config>.toml
osmosis --json rollout validate configs/eval/<config>.toml
osmosis --json train submit configs/training/<config>.toml
osmosis --json eval run configs/eval/<config>.toml
osmosis --json train status <run-name>
```
