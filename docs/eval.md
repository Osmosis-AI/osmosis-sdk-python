# Eval

Run **`osmosis eval run`** with a TOML file to evaluate a rollout against a
dataset using the same controller protocol used by training. Results are cached
on disk so long runs can resume after interruption.

Eval configs must live under `configs/eval/` inside a structured Osmosis
project.

## TOML configuration

### Required

**`[eval]`**

| Key | Description |
|-----|-------------|
| `rollout` | Rollout pack name; eval runs the entrypoint from `rollouts/<rollout>/`. |
| `entrypoint` | Python file started with `uv run python <entrypoint>` from the rollout directory. |
| `dataset` | Dataset file path, relative to the project root. |

**`[llm]`**

| Key | Description |
|-----|-------------|
| `model` | Model id in LiteLLM provider/model form, such as `openai/gpt-5-mini` or `hosted_vllm/Qwen/Qwen2.5-7B-Instruct`. |
| `base_url` | Optional OpenAI-compatible API base for the selected LiteLLM provider. |
| `api_key_env` | Optional environment variable name holding the API key; omit only for LiteLLM providers/endpoints that do not require credentials. |

### Optional sections

Eval configs use `[eval]`, `[llm]`, and optional `[runs]`, `[timeouts]`, and
`[output]` settings. Legacy `[grader]` and `[baseline]` sections are rejected:
grading is reported by the rollout server callback, and model comparisons should
use separate eval configs.

**`[runs]`**

| Key | Default | Description |
|-----|---------|-------------|
| `n` | `1` | Runs per row (pass@k). |
| `batch_size` | `1` | Concurrent runs. |
| `pass_threshold` | `1.0` | Score >= threshold counts as pass. |

**`[timeouts]`**

| Key | Default | Description |
|-----|---------|-------------|
| `agent_workflow_timeout_s` | `450` | Max seconds to wait for the rollout completion callback. |
| `grader_timeout_s` | `150` | Max seconds to wait for the grader completion callback. |

**`[output]`**

| Key | Default | Description |
|-----|---------|-------------|
| `log_samples` | `false` | Persist full message logs to JSONL alongside the cache. |
| `output_path` | - | Structured results directory (CLI `-o` overrides). |
| `quiet` | `false` | Less console output. |
| `debug` | `false` | Debug logging / traces. |

**`[eval]` extras**

| Key | Default | Description |
|-----|---------|-------------|
| `limit` | - | Max rows (CLI `--limit` overrides when set). |
| `offset` | `0` | Skip first N rows. |
| `fresh` | `false` | Start fresh (same as CLI `--fresh`). |
| `retry_failed` | `false` | Only failed runs (same as CLI `--retry-failed`). |

### Example `configs/eval/my-rollout.toml`

```toml
[eval]
rollout = "my_rollout"
entrypoint = "main.py"
dataset = "data/my-eval.jsonl"

[llm]
model = "openai/gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[runs]
n = 1
batch_size = 1
pass_threshold = 1.0

[timeouts]
agent_workflow_timeout_s = 450
grader_timeout_s = 150

[output]
log_samples = false
```

## Quick start

```bash
osmosis eval run configs/eval/my-rollout.toml
osmosis eval run configs/eval/my-rollout.toml --fresh
osmosis eval run configs/eval/my-rollout.toml --limit 50 --batch-size 4 -o ./results
```

## Local eval setup

### Controller-Backed Local Eval

`osmosis eval run` starts your rollout as a local HTTP server and drives it through the same controller protocol used by training. Eval runs `uv run python <entrypoint>` from `rollouts/<rollout>`, expects the server to bind `127.0.0.1:8000` or `0.0.0.0:8000`, requires `GET /health`, sends `POST /rollout`, serves model calls from the controller at `POST /chat/completions`, and waits for the rollout to call `POST /v1/rollout/completed` plus `POST /v1/grader/completed` callback URLs.

Eval configs no longer contain `[grader]` or `[baseline]`. Grading is part of the rollout server. To compare two models, run two eval configs separately.

Use `osmosis eval run configs/eval/<name>.toml --limit 1` for the end-to-end smoke test that covers server startup, `/health`, `/rollout`, `/chat/completions`, callbacks, provider credentials, and grading.

## Rollout server contract

The rollout entrypoint must start a server built with `create_rollout_server(...)`.
For local eval, the CLI provides `ROLLOUT_PORT=8000` and runs the entrypoint from
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
2. Cache files live under project-local `.osmosis/cache/eval/`.
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
