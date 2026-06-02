# Eval

An evaluation run submits against a platform dataset using the same workspace, rollout, entrypoint, dataset, and optional `commit_sha` semantics as `osmosis train submit`. The platform clones the repository identified by the workspace directory's `origin` remote and executes the rollout server-side, so push your changes and confirm Git Sync before submitting.

Evaluation configs must live under `configs/eval/` inside a structured Osmosis workspace directory.

## TOML configuration

### Required fields

**`[experiment]`**

| Key | Description |
|-----|-------------|
| `rollout` | Directory name under `rollouts/`. |
| `entrypoint` | Python file relative to the rollout directory. |
| `model_path` | LiteLLM-style model name for the evaluation policy model, such as `openai/gpt-5-mini`. |
| `dataset` | Platform dataset name from `osmosis dataset list`. |

### Optional fields and sections

Evaluation submit configs also support optional `[experiment].commit_sha`, `[evaluation]`, and `[env]`. They must include `[secrets].required`; default OpenAI eval configs should include `OPENAI_API_KEY`, and `required = []` is only for evaluations that need no secret refs. The SDK validates only shallow TOML shape, required fields, recognized keys, and env-var names; backend validation owns provider, dataset, model, and evaluation parameter errors.

| Key | Description |
|-----|-------------|
| `commit_sha` | Optional pinned commit. When omitted, the platform chooses source from the connected repository. |

**`[evaluation]`**

| Key | Description |
|-----|-------------|
| `limit` | Optional row cap. |
| `n` | Number of evaluation attempts. |
| `batch_size` | Rows evaluated per batch. |
| `pass_threshold` | Minimum passing score. |
| `agent_workflow_timeout_s` | Agent workflow timeout per row. |
| `grader_timeout_s` | Grader timeout per row. |

**`[env]` / `[secrets]`**

`[env]` contains literal env-var values. `[secrets].required` lists platform `environment_secret` record names; the platform resolves each to its encrypted value server-side and injects it as an env var of the same name. `[env]` keys must match `^[A-Z_][A-Z0-9_]*$`; secret names must match `^[A-Z][A-Z0-9_]*$`; `[env]` names must not start with `_OSMOSIS_`; and a name cannot appear in both `[env]` and `[secrets].required`. Register secrets with `osmosis secret set`; personal scope is the default, and `--scope workspace` creates workspace-shared secrets.

### Example `configs/eval/my-rollout.toml`

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

[secrets]
# Default OpenAI eval models need this platform secret.
# Use [] only when this evaluation needs no secret refs.
required = ["OPENAI_API_KEY"]
```

## Quick start

```bash
osmosis dataset list
git push                                       # ensure the platform sees your commit
osmosis eval submit configs/eval/my-rollout.toml
```

## Rubric (LLM-as-judge)

`osmosis eval rubric` is a local utility for scoring an existing JSONL conversation file with an LLM judge. It does not require a workspace directory or platform authentication, and it does not run a rollout.

```bash
osmosis eval rubric -d conversations.jsonl \
  --rubric "Evaluate the assistant's helpfulness..." \
  --model openai/gpt-5-mini
```

See the [CLI reference](./cli.md#osmosis-eval-rubric) for the full flag list.

## See also

- [Dataset format](./datasets.md)
- [CLI reference](./cli.md)
- [Troubleshooting](./troubleshooting.md)
