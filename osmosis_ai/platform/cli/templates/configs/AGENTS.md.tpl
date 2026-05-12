# Training & Evaluation Configs

Configs are project-scoped and must stay in their canonical directories.

## Canonical paths

- Training: `configs/training/<name>.toml`
- Eval: `configs/eval/<name>.toml`

Do not place these configs elsewhere. The CLI validates these locations.
For AI agents or automation, prefer `osmosis --json ...` for structured output
or `osmosis --plain ...` for low-noise text.

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

### Environment variables and secrets (optional)

Add `[rollout.env]` and/or `[rollout.secrets]` sections to inject environment
variables into the rollout container at training time.

```toml
[rollout.env]
# Literal values — visible in this file. Do NOT put secrets here.
LOG_LEVEL = "INFO"
MY_CONFIG = "some-value"

[rollout.secrets]
# Value is the *name* of a Platform environment_secret record, not the secret
# value itself. The platform resolves and injects it server-side.
# Pre-register the secret at /:orgName/secrets before submitting.
OPENAI_API_KEY = "openai-api-key"
```

Rules:
- Keys must match `^[A-Z_][A-Z0-9_]*$`
- The same key cannot appear in both sections
- Reserved names (managed by the platform) are forbidden:
  `GITHUB_CLONE_URL`, `GITHUB_TOKEN`, `ENTRYPOINT_SCRIPT`, `REPOSITORY_PATH`,
  `TRAINING_RUN_ID`, `ROLLOUT_NAME`, `ROLLOUT_PORT`

Inside the rollout container both sets of vars are available via `os.environ`.

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
osmosis project doctor
osmosis eval run configs/eval/<config>.toml --limit 1
osmosis train submit configs/training/<config>.toml
osmosis eval run configs/eval/<config>.toml
osmosis train status <run-name>
```
