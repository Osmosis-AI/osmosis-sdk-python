# Test mode

Run your **AgentWorkflow** against a dataset using an external LLM via [LiteLLM](https://docs.litellm.ai/docs/providers). This is a **smoke test**: successes and failures only, unless your entrypoint also defines a grader (unusual for quick tests).

Implementation-wise, `osmosis rollout test` delegates to the same engine as `osmosis eval run` with a generated TOML file.

The command is currently hidden from standard help output, so you may need to invoke it directly by name as `osmosis rollout test ...`.

## Project layout

The generated config uses rollout name `_rollout_test` and entrypoint `workflow.py`. That implies:

1. A file **`workflow.py`** in the **current working directory** (or importable on `sys.path`) containing exactly one concrete `AgentWorkflow` subclass.
2. A directory **`rollouts/_rollout_test/`** (may be empty). The CLI prepends it to `sys.path` so imports from that pack resolve consistently with `osmosis eval run`.

If either is missing, loading fails with a clear CLI error.

## Quick start

```bash
mkdir -p rollouts/_rollout_test
osmosis rollout test -m _ -d data.jsonl --model gpt-5-mini
osmosis rollout test -m _ -d data.jsonl --model openai/gpt-5-mini --limit 10
```

`-m` / `--module` is **required** by the CLI today; use any placeholder (e.g. `_`). The workflow is always resolved from `workflow.py` as above.

Optional:

```bash
osmosis rollout test -m _ -d data.jsonl --model gpt-5-mini --base-url https://api.openai.com/v1
osmosis rollout test -m _ -d data.jsonl --model gpt-5-mini --api-key sk-...
```

## CLI reference

```
osmosis rollout test [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--module`, `--agent` | (required) | Placeholder module flag (see above). |
| `-d`, `--dataset` | (required) | Dataset path (`.parquet` recommended, `.jsonl`, `.csv`). |
| `--model` | `gpt-5-mini` | Model id; if no `/`, prefixed as `openai/<model>`. |
| `--api-key` | env | API key for the LLM call. |
| `--base-url` | — | OpenAI-compatible base URL. |
| `--limit` | all | Max rows. |
| `--offset` | `0` | Rows to skip. |
| `-q`, `--quiet` | off | Less output. |
| `--debug` | off | Verbose / trace-friendly logging. |

## When to use `osmosis eval run` instead

For graded metrics, pass@k, baselines, structured output, or explicit rollout packs, use a checked-in TOML and **`osmosis eval run`** — see [Eval mode](./eval-mode.md).

## Interactive stepping

Step-by-step debugging is not exposed on `rollout test` in the current CLI. Use `--debug` and smaller `--limit`, or integrate tests in Python against `osmosis_ai.rollout_v2` types directly.

## Exceptions

Shared with eval: see [Eval mode — Exceptions](./eval-mode.md#exceptions) and [Troubleshooting](./troubleshooting.md).

## See also

- [Eval mode](./eval-mode.md)
- [Dataset format](./datasets.md)
- [CLI reference](./cli.md)
