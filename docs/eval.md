# Eval

Submit **`osmosis eval submit`** with a TOML file to run a cloud eval against a
platform dataset. Eval submission uses the same workspace, rollout, entrypoint,
dataset, and optional `commit_sha` semantics as `osmosis train submit`.

Eval configs must live under `configs/eval/` inside a structured Osmosis
workspace directory.

## TOML configuration

### Required fields

**`[experiment]`**

| Key | Description |
|-----|-------------|
| `rollout` | Rollout name. |
| `entrypoint` | Python file relative to the rollout directory. |
| `dataset` | Platform dataset name from `osmosis dataset list`. |

**`[llm]`**

| Key | Description |
|-----|-------------|
| `model_path` | LiteLLM-style model name for the eval policy model, such as `openai/gpt-5-mini`. |

### Optional fields and sections

Eval submit configs also support optional `[experiment].commit_sha`,
`[llm].base_url`, `[evaluation]`, `[env]`, and `[secrets]`. The SDK validates
only shallow TOML shape, required fields, recognized keys, and env-var names;
backend validation owns provider, dataset, model, and evaluation parameter
errors.

| Key | Description |
|-----|-------------|
| `commit_sha` | Optional pinned commit. When omitted, the platform chooses source from the connected repository. |
| `base_url` | Optional LiteLLM/OpenAI-compatible API base URL, such as `https://api.openai.com/v1`. Unlike evaluation params, no default is applied when this is omitted. |

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

`[env]` contains literal env-var values. `[secrets]` maps env-var names to
workspace secret record names resolved server-side. Keys must match
`^[A-Z_][A-Z0-9_]*$`, must not start with `_OSMOSIS_`, and cannot appear in both
sections.

### Example `configs/eval/my-rollout.toml`

```toml
[experiment]
rollout = "my_rollout"
entrypoint = "main.py"
dataset = "my-platform-dataset"
# commit_sha =

[llm]
model_path = "openai/gpt-5-mini"      # LiteLLM-style model name
# Optional LiteLLM/OpenAI-compatible base URL; no default is applied when omitted.
# base_url = "https://api.openai.com/v1"

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

## Quick start

```bash
osmosis dataset list
osmosis eval submit configs/eval/my-rollout.toml
```

## Local eval setup

`osmosis eval run` remains available during the local eval deprecation window and
still uses the legacy local eval config shape. Future work will remove local eval
from the public CLI surface.

### Controller-Backed Local Eval

`osmosis eval run` starts your rollout as a local HTTP server and drives it through the same controller protocol used by training. Eval runs `uv run python <entrypoint>` from `rollouts/<rollout>`, expects the server to bind `127.0.0.1:8000` or `0.0.0.0:8000`, requires `GET /health`, sends `POST /rollout`, serves model calls from the controller at `POST /chat/completions`, and waits for the rollout to call `POST /v1/rollout/completed` plus `POST /v1/grader/completed` callback URLs.

Eval configs no longer contain `[grader]` or `[baseline]`. Grading is part of the rollout server. To compare two models, run two eval configs separately.

Use `osmosis eval run configs/eval/<name>.toml --limit 1` for the end-to-end smoke test that covers server startup, `/health`, `/rollout`, `/chat/completions`, callbacks, provider credentials, and grading.

## Rollout server contract

The rollout entrypoint must start a server built with `create_rollout_server(...)`.
For local eval, the CLI provides `_OSMOSIS_ROLLOUT_PORT=8000` and runs the entrypoint from
the rollout directory.

The rollout server must expose:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Server readiness check. |
| `POST /rollout` | Controller request that starts one sample run. |

The `/rollout` payload includes these controller URLs:

| Payload field | Purpose |
|---------------|---------|
| `chat_completions_url` | Base URL for the controller-hosted OpenAI-compatible model endpoint. Clients call `<chat_completions_url>/chat/completions`. |
| `completion_callback_url` | Full callback URL the rollout calls when agent work finishes. |
| `grader_callback_url` | Full callback URL the rollout calls with grader rewards. |

The controller endpoints are not implemented by the rollout server; they are
hosted by the local eval process and passed to the rollout in the `/rollout`
request.

Each rollout directory must contain `pyproject.toml` so `uv run python
<entrypoint>` can start from `rollouts/<rollout>`.

## Connecting to model endpoints

Use `[llm].base_url` for OpenAI-compatible servers. The `model` still uses
LiteLLM provider/model form; for no-auth local servers, omit `api_key_env`.

| Serving | Example `model` | Example `base_url` |
|---------|-----------------|--------------------|
| vLLM | `hosted_vllm/Qwen/Qwen2.5-7B-Instruct` | `http://localhost:8001/v1` |
| SGLang | `hosted_vllm/Qwen/Qwen2.5-7B-Instruct` | `http://localhost:30000/v1` |
| Ollama | `ollama_chat/llama3.1` | `http://localhost:11434` |
| Osmosis inference | Provider/model id for your deployment | URL provided for your deployment |

## pass@k

When `[runs].n` > 1, each row is executed multiple times. Summary output
includes estimated pass@k values derived from pass/fail counts vs
`pass_threshold`.

## Result caching and resume

1. A **task id** is derived from the effective configuration, dataset
   fingerprint, rollout filesystem fingerprint, entrypoint path, and controller
   protocol version.
2. Cache files live under workspace-directory local `.osmosis/cache/eval/`.
3. Re-running the **same** command resumes an in-progress cache; use `--fresh`
   to discard.

**Environment:**

| Variable | Description |
|----------|-------------|
| `OSMOSIS_EVAL_LOCK_TIMEOUT` | Lock acquisition timeout seconds (default `30`). |

Cache invalidation includes TOML and CLI overrides that affect result semantics,
dataset content, controller protocol version, and rollout files with these
suffixes: `.py`, `.toml`, `.json`, `.jsonl`, `.yaml`, and `.yml`. Changes in
other rollout assets or external dependencies may not invalidate the cache; use
`--fresh` after changing them or upgrading libraries.

### Cache CLI

```bash
osmosis eval cache ls
osmosis eval cache rm <task_id>
osmosis eval cache rm --all --yes
```

## CLI overrides

Flags on `osmosis eval run` override TOML when provided:

| Flag | Effect |
|------|--------|
| `--fresh` | Discard cached results for this config |
| `--retry-failed` | Re-run failures only (mutually exclusive with `--fresh`) |
| `--limit` / `--offset` | Row window |
| `--batch-size` | Concurrency |
| `-o` / `--output-path` | Results directory |
| `--log-samples` | Save transcripts |
| `-q` / `--quiet` | Quiet |
| `--debug` | Debug |

## Exceptions

| Kind | Typical cause |
|------|----------------|
| CLI / `CLIError` | Bad TOML, missing API key env, rollout server startup errors, callback or grading failures |
| Failed eval run | Rollout or grader callback did not arrive before its configured timeout |
| `TimeoutError` | Cache lock held by another process or server startup health wait timed out |
| `RuntimeError` | Cache version mismatch, hash collision, dataset changed mid-run |

Dataset and provider errors are shared with [Troubleshooting](./troubleshooting.md).

## See also

- [Dataset format](./datasets.md)
- [CLI reference](./cli.md)
- [Troubleshooting](./troubleshooting.md)
