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
Manage projects, repositories, secrets, and account settings in the Osmosis
Platform product.

## Project Flow

Create the project in the Osmosis Platform, clone the repository created there,
then run CLI commands from that checkout.

```bash
git clone <repo-url>
cd <repo>
osmosis auth login
osmosis project doctor
osmosis template apply multiply              # or add your rollout under rollouts/
cp configs/training/default.toml configs/training/<run>.toml
$EDITOR configs/training/<run>.toml          # set rollout, dataset, and model_path
git add rollouts configs data research
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the checkout's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for repo-scoped commands.

For CI:

```bash
export OSMOSIS_TOKEN=<token>
osmosis train submit configs/training/<run>.toml --yes
```

### osmosis project doctor

```bash
osmosis project doctor
osmosis project doctor ./path/to/project
osmosis project doctor --fix --yes
```

Inspect and optionally repair the scaffold in the current Git checkout. Without
`--fix`, the command reports the project root, Git identity, required scaffold
paths, and missing paths. Add `--fix --yes` to create missing scaffold paths and
refresh agent scaffold files without prompting.

## Rollout

### osmosis rollout list

List rollouts for the current project repository.

```bash
osmosis rollout list
osmosis rollout list --limit 50
osmosis rollout list --all
```

## Evaluation

### osmosis eval run

Evaluate using a TOML config. Controller-backed eval starts the configured
rollout entrypoint as a local HTTP server with `uv run python <entrypoint>` from
`rollouts/<rollout>/`, sends `POST /rollout`, provides model calls through the
controller's `/chat/completions` endpoint, and waits for rollout and grader
callback URLs.

`osmosis eval run` expects the config file to live under `configs/eval/` inside a
structured Osmosis project. Eval configs use `[eval]`, `[llm]`, `[runs]`,
`[timeouts]`, and `[output]`; `[grader]` and `[baseline]` are no longer
supported.

```bash
osmosis eval run configs/eval/my-rollout.toml --limit 1
osmosis eval run configs/eval/my-rollout.toml
osmosis eval run configs/eval/my-rollout.toml --fresh
osmosis eval run configs/eval/my-rollout.toml --retry-failed
osmosis eval run configs/eval/my-rollout.toml --limit 20 --batch-size 4
osmosis eval run configs/eval/my-rollout.toml -o ./results --log-samples
```

See [Eval](./eval.md) for the full `[eval]`, `[llm]`, `[runs]`, `[timeouts]`,
and `[output]` sections.

### osmosis eval cache

```bash
osmosis eval cache ls
osmosis eval cache ls --model gpt-4 --status completed
osmosis eval cache rm <task_id>
osmosis eval cache rm --all --yes
```

| Subcommand / option | Description |
|---------------------|-------------|
| `ls` | List caches (`--model`, `--dataset`, `--status`) |
| `rm` | Delete by `task_id`, `--all`, or filters (`-y` skips prompt) |

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

## Training

### osmosis train submit

Submit a training run from a TOML config.

```bash
osmosis train submit configs/training/my-run.toml
osmosis train submit configs/training/my-run.toml --yes   # skip confirmation
```

The config file must live under `configs/training/` inside a structured Osmosis
project. The CLI reads the config locally and sends it to the platform, which
clones the repository identified by the checkout's `origin` remote for the
actual rollout code.
`osmosis train submit` includes the training preflight checks before launch; run
`osmosis eval run configs/eval/<name>.toml --limit 1` first when you want an
end-to-end local smoke test of the rollout server and grader.

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

#### Environment variables — `[rollout.env]`

Literal key/value pairs injected verbatim into the rollout container. Values
are visible in the config file and in CLI output — do **not** use this section
for secrets.

```toml
[rollout.env]
LOG_LEVEL = "INFO"
DEFAULT_REGION = "us-west-2"
```

#### Secrets — `[rollout.secrets]`

Maps env-var names to Platform `environment_secret` **record names**. The
platform resolves the actual secret value server-side from encrypted secret
storage and injects it into the container. Secret values never
appear in the config file, in the API payload, or in CLI output.

Pre-register secrets at `/:orgName/secrets` in the platform UI before
submitting a run that references them.

```toml
[rollout.secrets]
OPENAI_API_KEY = "openai-api-key"   # "openai-api-key" is the record name
```

#### Rules for both sections

- Keys must match `^[A-Z_][A-Z0-9_]*$`.
- A key cannot appear in both `[rollout.env]` and `[rollout.secrets]`.
- Reserved names that cannot be used (managed by the platform):
  `GITHUB_CLONE_URL`, `GITHUB_TOKEN`, `ENTRYPOINT_SCRIPT`, `REPOSITORY_PATH`,
  `TRAINING_RUN_ID`, `ROLLOUT_NAME`, `ROLLOUT_PORT`.
- Both sections are optional.

### osmosis train status

Show details for a training run.

```bash
osmosis train status <run-name>
osmosis --json train status <run-name>
```

### osmosis train list

List training runs for the current project repository.

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

### osmosis train metrics

Export training run metrics to a JSON file.

```bash
osmosis train metrics <run-name>
osmosis train metrics <run-name> -o ./my-metrics.json
```

## Deployment

Deployments expose trained checkpoint adapters for inference. Use the checkpoint
UUID or checkpoint name anywhere `<checkpoint>` appears.

### osmosis deployment list

List deployments for the current project repository.

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
