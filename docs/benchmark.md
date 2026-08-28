# Benchmark

> Product/end-user benchmark usage (leaderboards, published scores, config field reference) lives at [docs.osmosis.ai](https://docs.osmosis.ai). This page is the **code-anchored** contract for `osmosis benchmark submit`: the TOML the SDK validates locally, how secret references resolve, and the submit flow. Download commands are covered in [run-downloads.md](./run-downloads.md); command-shell internals in [cli.md](./cli.md).

## `osmosis benchmark submit`

Reads a benchmark run config TOML, validates its **structure** locally, and POSTs it to the platform. Benchmark submit owns the config-specific front half because it fans out across multiple agents and carries extra secret references, then joins the shared secret-resolution and confirmation tail without source-fetch checks.

- Command shell: [../osmosis_ai/cli/commands/benchmark.py](../osmosis_ai/cli/commands/benchmark.py) (`benchmark_submit`)
- Handler: [../osmosis_ai/platform/cli/benchmark.py](../osmosis_ai/platform/cli/benchmark.py) (`submit`, `_submit_benchmark`)
- Config loader: [../osmosis_ai/platform/cli/benchmark_config.py](../osmosis_ai/platform/cli/benchmark_config.py) (`load_benchmark_submit_config`)
- Shared config primitives: [../osmosis_ai/platform/cli/shared_config.py](../osmosis_ai/platform/cli/shared_config.py)

```bash
osmosis benchmark submit configs/benchmark/<name>.toml        # interactive confirmation
osmosis benchmark submit configs/benchmark/<name>.toml --yes  # skip the prompt
osmosis benchmark submit configs/benchmark/<name>.toml --secrets-file .env.run
osmosis --workspace <name> benchmark submit /any/path/<name>.toml --yes
```

The config may live at any readable path. Root `--workspace <name>` lets the command run without a local Git repository or Osmosis scaffold; without it, the CLI preserves the existing current-directory Git scope. Get a benchmark key from `osmosis benchmark list` (the shell-safe `Key` column, e.g. `terminal-bench-2-1`).

### Config contract (what the SDK validates)

```toml
[experiment]                        # required
benchmark = "terminal-bench-2-1"

[tasks]                             # optional; omitted runs every task
task_set = "parity"

[[agents]]                          # required, 1-8 entries
harness = "mini-swe-agent"

[agents.model]
type = "provider"
model = "anthropic/claude-sonnet-4"
api_key_secret = "ANTHROPIC_API_KEY"

[agents.env]                        # optional, per agent; overrides [env]
LOG_LEVEL = "debug"

[execution]                         # optional; structure-only, values owned by backend
attempts_per_task = 3
judge_model = "openai/gpt-5"
judge_api_key_secret = "OPENAI_API_KEY"

[verifier]                          # optional
required = ["VLM_API_KEY"]

[secrets]                           # optional
required = ["MY_TOOL_TOKEN"]

[env]                               # optional, applies to every agent
LOG_LEVEL = "info"
```

| Section | Required | SDK enforces locally |
|---------|----------|----------------------|
| `[experiment]` | yes | Only `benchmark`, the benchmark key (`BenchmarkExperimentSection`). |
| `[tasks]` | no | `categories` and `task_names` are lists of non-blank strings; `task_set` accepts only `"parity"`. Omitting the section selects every task. |
| `[[agents]]` | yes | An array of tables with 1-8 entries; each has `harness`, optional `harness_api_key_secret` (`cursor-cli` only), `model` (see [Agent models](#agent-models)), and `env` (`BenchmarkAgentSection`). |
| `[execution]` | no | Structure only — known keys are `attempts_per_task`, `max_concurrent_attempts`, `timeout_multiplier`, `max_retries`, `pass_threshold`, `judge_model`, `judge_api_key_secret`; unknown keys rejected, **values forwarded unvalidated**. |
| `[verifier]` | no | Only `required`, up to 16 secret record names the dataset's verifier reads. Forwarded to the platform as `execution.verifier_secrets`. |
| `[secrets]` | no | Only `required`, up to 16 names — the only references that may be supplied per run. |
| `[env]` | no | Names match `ENV_VAR_NAME_RE` = `^[A-Z_][A-Z0-9_]*$`, must not start with `_OSMOSIS_` (reserved), values must be strings. Applies to every agent; a name repeated in `[agents.env]` overrides it for that agent. |

Every section rejects unknown keys (`extra="forbid"`), so a typo fails locally rather than being silently forwarded.

`task_set` is deliberately strict: the platform route silently expands an unrecognized `task_set` to the **full benchmark**, so a typo there would run every task instead of erroring.

### Agent models

`[agents.model]` is a discriminated union on `type`; each variant requires different keys.

| `type` | Required keys | Notes |
|--------|---------------|-------|
| `"provider"` | `model`, `api_key_secret` | A hosted provider model, e.g. `anthropic/claude-sonnet-4`. |
| `"endpoint"` | `base_url`, `model`, `api_key_secret` | Optional `extra_headers`; an `Authorization` header is **rejected** — authenticate with `api_key_secret` instead. |
| `"hosted"` | `base_model`, `lora_model_name` | A model hosted on the platform; it references no secret. |

### Secret references

A benchmark config can reference secrets from five places. `BenchmarkSubmitConfig.required_secrets` collects them in order, deduplicated: each agent's model `api_key_secret`, each `harness_api_key_secret`, `execution.judge_api_key_secret`, `[verifier].required`, then `[secrets].required`. Every name must match `SECRET_NAME_RE` = `^[A-Z][A-Z0-9_]*$`.

Only `[secrets].required` names may be supplied at submit time, via `--secrets-file`, the process environment, or a prompt — the resolution order is the shared one described in [Secret resolution](./eval.md#secret-resolution). Every other reference must **already exist** as a stored record in the workspace or personal scope; submit fails fast on anything missing with an `osmosis secret set <name>` hint. The confirmation table labels each name with its scope, or `Run` for a per-run value.

Three rules exist because the platform injects a referenced secret's value as an env var of the **same name** into that agent's runtime env:

- **Env collisions are rejected.** A literal env var with the same name as a referenced secret would be silently overwritten, so submit errors instead. The check is scoped per agent against the effective env (`[env]` merged with that agent's `[agents.env]`), so one agent's secret name may still be another agent's literal env var.
- **Some model secret names are reserved.** `api_key_secret` cannot be `DAYTONA_API_KEY`, `DAYTONA_API_URL`, `SKYPILOT_SERVICE_ACCOUNT_TOKEN`, or `SKYPILOT_API_SERVER_ENDPOINT` — the runner removes those before model-key aliasing.
- **Harness credentials travel a separate channel.** `cursor-cli` reads `CURSOR_API_KEY` and must set `harness_api_key_secret` to a record named *exactly* that variable. Mini SWE-agent rejects `harness_api_key_secret`: for provider and endpoint models, the platform injects the model's `api_key_secret` as `MSWEA_API_KEY`, so that name cannot also be a literal env var; hosted models receive no injected model key and may set `MSWEA_API_KEY` explicitly.

### SDK vs backend validation

As with eval, the SDK validates **structure only**; the backend owns value-level semantics. The SDK does not check benchmark or task-name existence, model or provider validity, or execution parameter ranges — `[execution]` values are typed as passthrough and forwarded unvalidated. Those errors surface from the platform at submit time.

### Submit flow (what happens locally before the POST)

`submit` ([benchmark.py](../osmosis_ai/platform/cli/benchmark.py)) runs in order:

1. Resolve explicit root `--workspace` scope, or fall back to the current Git workspace.
2. Resolve the config path without imposing a scaffold-directory containment rule.
3. Load + validate the TOML (`load_benchmark_submit_config`), including env-var names and all secret-reference rules.
4. Render the run summary, per-agent model table, and env table. The displayed `attempts_per_task` and `max_concurrent_attempts` fall back to `1` and `4`, mirroring the route's defaults.
5. Warn with code `HLE_PARITY_RECOMMENDED` when the benchmark is HLE and `task_set` is not `"parity"`, since published HLE scores are parity-based. The name match is case-insensitive so a casing typo still surfaces the guidance, and the route's own case-sensitive error follows.
6. If any secret is referenced, fetch workspace + personal scopes, resolve `[secrets]` names, and **fail fast** on names that are neither stored nor supplied. Non-interactive resolution raises `INTERACTIVE_REQUIRED` with `details.missing` and `details.flags = ["--secrets-file"]`. If the scope lookup itself fails, names render without a scope rather than blocking the submit — the server still validates.
7. Confirm (skipped with `--yes`), then POST via `client.submit_benchmark_run`. Outside an interactive rich session, a missing `--yes` raises `INTERACTIVE_REQUIRED`; its details include `prompt` and `summary`. A missing-secret error from the platform is enriched with the same add-secret hint.

The result is an `OperationResult` whose next-steps point at `osmosis benchmark runs info <name>`, `osmosis benchmark runs list`, and the platform URL.

### Companion commands

Benchmarks and their runs are split across two namespaces: top-level `osmosis benchmark list|info` act on benchmarks, while `osmosis benchmark runs …` acts on your runs ([../osmosis_ai/platform/cli/benchmark.py](../osmosis_ai/platform/cli/benchmark.py)).

- `osmosis benchmark list [--all] [--limit N]` — available benchmarks, with the shell-safe `Key` to pass to `info` and `submit`.
- `osmosis benchmark info <key> [--all] [--limit N]` — one benchmark's metadata, task options, leaderboard, and runs. JSON output includes `requires_judge_api_key`.
- `osmosis benchmark runs list|info|logs|stop` — your runs for the workspace.
- `osmosis benchmark runs download …` — run outputs; see [run-downloads.md](./run-downloads.md) for the layout, resume, and confirmation behavior.

## See also

- [eval.md](./eval.md) — the `osmosis eval submit` contract and the shared secret-resolution order
- [run-downloads.md](./run-downloads.md) — benchmark and eval download commands
- [cli.md](./cli.md) — CLI internals (command shells, lazy imports, JSON envelopes)
