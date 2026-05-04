# Eval

Run **`osmosis eval run`** with a TOML file to execute an **AgentWorkflow** against a dataset using an external or self-hosted LLM. A **Grader** in the same entrypoint module provides reward scores.

Results are cached on disk so long runs can resume after interruption.

Eval configs must live under `configs/eval/` inside a structured Osmosis
project.

## TOML configuration

### Required

**`[eval]`**

| Key | Description |
|-----|-------------|
| `rollout` | Rollout pack name; the CLI adds `rollouts/<rollout>/` to `sys.path`. |
| `entrypoint` | Path to the Python file (e.g. `workflow.py`) that defines the workflow. |
| `dataset` | Path to the dataset file. |

**`[llm]`**

| Key | Description |
|-----|-------------|
| `model` | Model id in LiteLLM form (e.g. `openai/gpt-5-mini`) or the model name expected by the OpenAI-compatible endpoint configured in `[llm].base_url`. |
| `base_url` | Optional OpenAI-compatible API base. |
| `api_key_env` | Optional environment variable name holding the API key. |

### Optional sections

Most eval configs only need `[eval]`, `[llm]`, and optional `[runs]` / `[output]` settings. You do **not** typically need a `[grader]` table because the grader is usually auto-discovered from the entrypoint module.

**`[grader]`** (optional — when the grader lives outside the entrypoint)

| Key | Description |
|-----|-------------|
| `module` | Import path to the `Grader` class, e.g. `my_rollout.grader:MyGrader`. |
| `config` | Optional import path to the `GraderConfig` object, e.g. `my_rollout.grader:grader_config`. |

**`[runs]`**

| Key | Default | Description |
|-----|---------|-------------|
| `n` | `1` | Runs per row (pass@k). |
| `batch_size` | `1` | Concurrent runs. |
| `pass_threshold` | `1.0` | Score ≥ threshold counts as pass. |

**`[output]`**

| Key | Default | Description |
|-----|---------|-------------|
| `log_samples` | `false` | Persist full message logs to JSONL alongside the cache. |
| `output_path` | — | Structured results directory (CLI `-o` overrides). |
| `quiet` | `false` | Less console output. |
| `debug` | `false` | Debug logging / traces. |

**`[baseline]`** (optional — compare two models)

| Key | Description |
|-----|-------------|
| `model` | Baseline model id. |
| `base_url` | Optional baseline base URL. |
| `api_key_env` | Optional env var for baseline key. |

**`[eval]` extras**

| Key | Default | Description |
|-----|---------|-------------|
| `limit` | — | Max rows (CLI `--limit` overrides when set). |
| `offset` | `0` | Skip first N rows. |
| `fresh` | `false` | Start fresh (same as CLI `--fresh`). |
| `retry_failed` | `false` | Only failed runs (same as CLI `--retry-failed`). |

### Example `configs/eval/my-rollout.toml`

```toml
[eval]
rollout = "my_rollout"
entrypoint = "workflow.py"
dataset = "data/my-eval.parquet"

[llm]
model = "openai/gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[runs]
n = 4
batch_size = 2
pass_threshold = 1.0

[output]
log_samples = true
```

## Quick start

```bash
osmosis eval run configs/eval/my-rollout.toml
osmosis eval run configs/eval/my-rollout.toml --fresh
osmosis eval run configs/eval/my-rollout.toml --limit 50 --batch-size 4 -o ./results
```

## Workflow and grader discovery

- The **workflow** class is the single concrete `AgentWorkflow` subclass in the entrypoint module (plus optional `AgentWorkflowConfig`).
- The **grader** is usually auto-discovered from that same module, along with an optional `GraderConfig`, so most users do not need to declare grader wiring in TOML.
- If the grader lives in another module, set `[grader].module` and optional `[grader].config` to override auto-discovery.
- A grader is **required**; if none is found, `osmosis eval run` exits with an error.

## Connecting to model endpoints

Use `[llm].base_url` for OpenAI-compatible servers:

| Serving | Example `base_url` |
|---------|-------------------|
| vLLM | `http://localhost:8000/v1` |
| SGLang | `http://localhost:30000/v1` |
| Ollama | `http://localhost:11434/v1` |
| Osmosis inference | Use the URL provided for your deployment |

## pass@k

When `[runs].n` > 1, each row is executed multiple times. Summary output includes estimated pass@k values derived from pass/fail counts vs `pass_threshold`.

## Result caching and resume

1. A **task id** is derived from the full configuration, dataset fingerprint, and source fingerprints for the entrypoint / grader modules.
2. Cache files live under project-local `.osmosis/cache/eval/`, or `$OSMOSIS_CACHE_DIR/eval/` when set.
3. Re-running the **same** command resumes an in-progress cache; use `--fresh` to discard.

**Environment:**

| Variable | Description |
|----------|-------------|
| `OSMOSIS_CACHE_DIR` | Override cache root (eval uses `<root>/eval/`; the command must still run inside an Osmosis project). |
| `OSMOSIS_EVAL_LOCK_TIMEOUT` | Lock acquisition timeout seconds (default `30`). |

Cache invalidation includes: TOML and CLI overrides, dataset content, and entrypoint/grader source fingerprints. Changes in **external** dependencies not imported as part of those modules may not invalidate the cache — use `--fresh` after upgrading libraries.

### Cache CLI

```bash
osmosis eval cache dir
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
| CLI / `CLIError` | Bad TOML, import errors, missing API key env, grader failures |
| `TimeoutError` | Cache lock held by another process |
| `RuntimeError` | Cache version mismatch, hash collision, dataset changed mid-run |

Dataset and provider errors are shared with [Troubleshooting](./troubleshooting.md).

## See also

- [Dataset format](./datasets.md)
- [CLI reference](./cli.md)
- [Troubleshooting](./troubleshooting.md)
