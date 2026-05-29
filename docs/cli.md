# CLI reference

Installing the SDK provides a lightweight CLI as `osmosis` (aliases: `osmosis_ai`, `osmosis-ai`). The CLI loads `.env` from the current working directory via `python-dotenv`.

## Authentication

Credentials are stored at `~/.config/osmosis/credentials.json`.

### osmosis auth login

Device code flow:

```bash
osmosis auth login
osmosis auth login --force
```

### osmosis auth logout

```bash
osmosis auth logout
osmosis auth logout -y
```

### osmosis auth whoami

```bash
osmosis auth whoami
```

`whoami` verifies the active credentials and reports the authenticated account.
Manage workspaces, repositories, secrets, and account settings in the Osmosis
Platform product.

## Workspace Directory Flow

Create or open a workspace in the Osmosis Platform, clone the repository created there,
then run CLI commands from that workspace directory.

```bash
git clone <repo-url>
cd <repo>
osmosis auth login
osmosis doctor
osmosis template apply multiply              # or add your rollout under rollouts/
cp configs/training/default.toml configs/training/<run>.toml
$EDITOR configs/training/<run>.toml          # set rollout, dataset, and model_path
git add rollouts configs data
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the workspace directory's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for commands scoped by the workspace directory.

For CI:

```bash
export OSMOSIS_TOKEN=<token>
osmosis train submit configs/training/<run>.toml --yes
```

### osmosis doctor

```bash
osmosis doctor
osmosis doctor ./path/to/workspace-directory
osmosis doctor --fix
```

Inspect and optionally repair the scaffold in the current workspace directory. Without
`--fix`, the command reports the workspace directory, Git identity, required scaffold
paths, and missing paths. Add `--fix` to create missing scaffold paths and
check for official scaffold file updates without overwriting local edits.

## Rollout

### osmosis rollout list

List rollouts for the current workspace directory.

```bash
osmosis rollout list
osmosis rollout list --limit 50
osmosis rollout list --all
```

## Evaluation

### osmosis eval submit

Submit an evaluation run using a TOML config under `configs/eval/`. Evaluation
run configs align with training run configs for rollout identity and platform dataset
selection: `[experiment].dataset` is a platform dataset name from
`osmosis dataset list`, not a local `data/*.jsonl` path.

```toml
[experiment]
rollout = "my-rollout"
entrypoint = "main.py"
model_path = "openai/gpt-5-mini"      # LiteLLM-style model name
dataset = "my-platform-dataset"
# commit_sha =

[evaluation]
# Optional. Omit values to use platform defaults.
# limit = 200
# n = 1
# batch_size = 1
# pass_threshold = 1.0
# agent_workflow_timeout_s = 450
# grader_timeout_s = 150

# [env]
# LOG_LEVEL = "INFO"

# [secrets]
# OPENAI_API_KEY = "openai-api-key"
```

```bash
osmosis eval submit configs/eval/my-rollout.toml
```

### osmosis eval rubric

LLM-as-judge on a JSONL conversation file:

```bash
osmosis eval rubric -d data.jsonl \
  --rubric "Evaluate the assistant's helpfulness..." \
  --model openai/gpt-5.4
```

| Flag | Description |
|------|-------------|
| `-d` / `--data` | JSONL path (required) |
| `-r` / `--rubric` | Inline rubric or `@file.txt` (required) |
| `--model` | Judge model, LiteLLM form (required) |
| `-n` / `--number` | Runs per record |
| `-o` / `--output` | JSON results path |
| `--api-key` | Judge API key |
| `--timeout` | Seconds |
| `--score-min` / `--score-max` | Score range |

## Dataset

### osmosis dataset upload

Upload a local dataset file to the current workspace directory's platform
project. The dataset name is derived from the file name without its extension.

```bash
osmosis dataset upload data.jsonl
osmosis dataset upload data.jsonl --yes
osmosis dataset upload data.jsonl --overwrite
```

| Flag | Description |
|------|-------------|
| `--yes` / `-y` | Skip the interactive confirmation prompt |
| `--overwrite` | Replace an existing dataset with the same derived name |

When `--overwrite` is used, the CLI first confirms the duplicate-name conflict
with the platform, then creates a replacement dataset record and soft-deletes
the old one. The platform may reject overwrites while the existing dataset is
still uploading, still processing, or used by an active training run.

## Training

### osmosis train submit

Submit a training run from a TOML config.

```bash
osmosis train submit configs/training/my-run.toml
osmosis train submit configs/training/my-run.toml --yes   # skip confirmation
```

The config file must live under `configs/training/` inside a structured Osmosis
workspace directory. The CLI reads the config locally and sends it to the platform, which
clones the repository identified by the workspace directory's `origin` remote for the
actual rollout code.
`osmosis train submit` includes the training run preflight checks before launch;
run `osmosis eval submit configs/eval/<name>.toml` first when you want an
evaluation run before a training run.

#### Required `[experiment]` fields

| Key | Description |
|-----|-------------|
| `rollout` | Directory name under `rollouts/` |
| `entrypoint` | Python file relative to the rollout directory |
| `model_path` | Supported base model path |
| `dataset` | Platform dataset name (`osmosis dataset list`) |

#### Optional `[training]` fields

All fields are optional. Omitted fields use platform defaults.

| Key | Type | Description |
|-----|------|-------------|
| `total_epochs` | int | Number of passes over the dataset |
| `n_samples_per_prompt` | int | Rollout samples generated per prompt (GRPO group size) |
| `rollout_batch_size` | int | Prompts rolled out per training step. **Controls how many rollouts run concurrently on the rollout server** — set this to avoid overwhelming the LLM inference engine. Default: 64. For a 32-row dataset with a remote rollout server, `8` or `32` is a safe starting point. |
| `max_prompt_length` | int | Token limit for prompt inputs |
| `max_response_length` | int | Token limit for model responses |
| `lr` | float | Learning rate |
| `agent_workflow_timeout_s` | float | Seconds osmosis waits for the rollout server to complete one rollout before marking it failed (default: 450 s). Increase for long-horizon agent tasks where each rollout can take several minutes. |
| `grader_timeout_s` | float | Seconds osmosis waits for the grader callback after rollout completes (default: 150 s). Increase if your grader runs expensive verification. |

> **Sizing `rollout_batch_size`**: the rollout server processes `rollout_batch_size × n_samples_per_prompt` concurrent LLM calls per step. Too high a value overwhelms the inference engine and causes all rollouts to timeout. A good rule of thumb: `rollout_batch_size ≤ 32` when using a remote MCP-based rollout server with a 35B+ model.

#### Optional `[sampling]` fields

| Key | Type | Description |
|-----|------|-------------|
| `rollout_temperature` | float | Sampling temperature (default: 1.0) |
| `rollout_top_p` | float | Top-p nucleus sampling (default: 1.0) |

#### Optional `[checkpoints]` fields

| Key | Type | Description |
|-----|------|-------------|
| `checkpoint_save_freq` | int | Save a checkpoint every N rollout steps |
| `eval_interval` | int | Run evaluation every N rollout steps |

#### Environment variables — `[env]`

Literal key/value pairs injected verbatim into the rollout container. Values
are visible in the config file and in CLI output — do **not** use this section
for secrets.

```toml
[env]
LOG_LEVEL = "INFO"
DEFAULT_REGION = "us-west-2"
```

#### Secrets — `[secrets]`

Maps env-var names to Platform `environment_secret` **record names**. The
platform resolves the actual secret value server-side from encrypted secret
storage and injects it into the container. Secret values never
appear in the config file, in the API payload, or in CLI output.

Pre-register secrets at `/:orgName/secrets` in the platform UI before
submitting a run that references them.

```toml
[secrets]
OPENAI_API_KEY = "openai-api-key"   # "openai-api-key" is the record name
```

#### Rules for both sections

- Keys must match `^[A-Z_][A-Z0-9_]*$`.
- A key cannot appear in both `[env]` and `[secrets]`.
- Any env var name starting with `_OSMOSIS_` is reserved by the platform and cannot be used.
- Both sections are optional.

### osmosis train info

Show details, checkpoints, and metrics for a training run.

```bash
osmosis train info <run-name>
osmosis --json train info <run-name>
osmosis train info <run-name> -o ./my-metrics.json
```

### osmosis train list

List training runs for the current workspace directory.

```bash
osmosis train list
osmosis train list --limit 50
osmosis train list --all
```

### osmosis train stop

Stop a pending or running training run.

```bash
osmosis train stop <run-name>
osmosis train stop <run-name> --yes
```

## Deployment

Deployments expose trained checkpoint adapters for inference. Use the checkpoint
UUID or checkpoint name anywhere `<checkpoint>` appears.

### osmosis deployment list

List deployments for the current workspace directory.

```bash
osmosis deployment list
osmosis deployment list --limit 50
osmosis deployment list --all
```

### osmosis deployment info

Show deployment details for a checkpoint.

```bash
osmosis deployment info <checkpoint>
osmosis --json deployment info <checkpoint>
```

### osmosis deploy

Deploy or reactivate a checkpoint.

```bash
osmosis deploy <checkpoint>
```

### osmosis undeploy

Transition a checkpoint deployment to inactive.

```bash
osmosis undeploy <checkpoint>
```

## See also

- [Eval](./eval.md)
- [Dataset format](./datasets.md)
- [Troubleshooting](./troubleshooting.md)
