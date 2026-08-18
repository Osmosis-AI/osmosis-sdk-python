# `osmosis eval run` — local evaluation

Runs an evaluation on your machine against the rollout server in your own workspace, using the same TOML, the same rollout entrypoint, and the same v0.3 HTTP/callback protocol as `osmosis eval submit`. Command shell: [`osmosis_ai/cli/commands/eval.py`](../osmosis_ai/cli/commands/eval.py); business logic: [`osmosis_ai/platform/cli/eval_run.py`](../osmosis_ai/platform/cli/eval_run.py); supervisor: [`osmosis_ai/eval/local/`](../osmosis_ai/eval/local/).

## Install

```bash
pip install "osmosis-ai[eval-run]"          # LocalBackend rollouts
pip install "osmosis-ai[eval-run,harbor]"   # Harbor (sandboxed) rollouts
```

A missing dependency is reported with that exact command, so you never have to guess which extra you are short of.

## Requirements

`eval run` requires being logged in (`osmosis auth login`) and run from the workspace directory. All LLM traffic is served by an in-process LiteLLM bridge on your machine — the same design as the hosted eval controller — so `experiment.model_path` is a LiteLLM model id (`openai/gpt-5-mini`, `anthropic/claude-sonnet-4-6`, …) and provider credentials resolve exactly as LiteLLM resolves them anywhere: from the process environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, provider `*_API_BASE` overrides, …) or from `--secrets-file`. Local OpenAI-compatible endpoints (a laptop vLLM or Ollama) work through the corresponding LiteLLM provider id and its environment variables.

Execution is local-only: the rollout entrypoint must build `LocalBackend`, or `HarborBackend` on Harbor's Docker environment (the host Docker runtime). Harbor's SkyPilot environment is not supported by `eval run`.

## Basic use

```bash
osmosis eval run configs/eval/my-rollout.toml --name my-local-run
```

The same command re-run resumes: work items with a durable terminal result are skipped, everything else runs again.

```bash
osmosis eval run configs/eval/my-rollout.toml --dataset-file data/dev.jsonl --rows 3,7,10-20
```

`--rows` selects dataset rows by their offset in the file, which is the cheapest way to smoke-test a rollout. `--dataset-file` runs a local file instead of the platform dataset named in `experiment.dataset`.

## Options

| Flag | Meaning |
|---|---|
| `--name` | Stable run name. Re-running the same name resumes. Omit for a one-off run with a generated name. |
| `--output`, `-o` | Run output root. Default `.osmosis/evals/`. |
| `--dataset-file` | Local dataset file instead of the platform dataset. |
| `--secrets-file` | Dotenv file supplying `[secrets]` values; `-` reads stdin. Values are never written to disk. |
| `--rows` | Dataset rows to run, e.g. `3,7,10-20`. Overrides `evaluation.limit`. |
| `--fresh` | Archive this name's existing results and start clean. |
| `--retry-failed` | Also re-run failed and skipped work items. Successes are kept. |
| `--max-in-flight` | Concurrent rollouts. Default: `evaluation.batch_size`, then the backend's advertised capacity, then 1. |
| `--yes`, `-y` | Skip the cost confirmation. |

Advanced: `--rollout-port` pins the rollout-server port and `--verbose` streams log lines to the terminal. Before any dispatch the run makes one probe completion against the configured model, so a bad model id or key fails once, loudly, instead of failing every row.

There is deliberately **no** `[local]` or `[execution]` config section: everything that differs between a local and a cloud run is a CLI flag, so one TOML means the same thing to both commands.

## Output layout

```text
.osmosis/evals/<run-name>/
  manifest.json        # local provenance + the resolved-input lock (never uploaded)
  events.jsonl         # terminal-result journal — the resume authority
  logs.txt             # combined supervisor + rollout-server log, secrets redacted
  index.jsonl          # platform-shaped sample index
  progress.json        # {total_runs, sampled_rows, total_dataset_rows}
  metrics.json         # {eval_run: {...}, summary: {...}}
  summary.jsonl        # index.jsonl, verbatim
  trajectories/row_<r>_run_<n>.json
  artifacts/row_<r>_run_<n>/...
  rollout_trials/<rollout-id>/
    trajectory.json    # canonical ATIF document
    diagnostics.json
    logs/              # Harbor's native per-trial logs, when the backend has them
    artifacts/...
```

`metrics.json`, `summary.jsonl`, `trajectories/`, and `artifacts/` are the same shape `osmosis eval download` produces for a cloud run, so the same tooling reads both. Everything under `rollout_trials/` is the canonical store; the top-level copies are independent, so editing them never corrupts the originals.

## Resume, fresh, and retry

Resume is **crash and interrupt recovery, not iteration recovery.**

A work item is complete only when it has a durable terminal result. A result is durable once its journal record has been written and `fsync`-ed, which happens *before* the rollout server's callback is acknowledged — so a `kill -9` can never lose an acknowledged result, and can never skip an unacknowledged one. Ctrl-C cancels in-flight rollouts without writing a terminal record, so they simply run again next time.

A named run is pinned to its resolved inputs: model, dataset bytes, selected rows, `n`, timeouts, pass threshold, rollout entrypoint, and a digest of the rollout's source files. Change any of them and the run refuses to resume, naming the input group that changed:

```text
Error: run 'my-local-run' was created with different resolved inputs, so resuming it would
mix versions inside one set of metrics. Changed: rollout. Restart under the same name with
--fresh (the previous results are archived, never deleted).
```

This is deliberate. The common loop — "some rows failed, edit the agent, re-run the failures" — would otherwise produce one set of metrics computed from two different versions of your code. Use `--fresh` to restart under the same name; the previous results move to `.osmosis/evals/<name>.archive-<utc-timestamp>/`.

`--retry-failed` re-runs failed and skipped items *without* a code change, which is the case where mixing versions cannot happen. Each attempt gets a fresh rollout id; the superseded attempt's directory stays for diagnosis but never appears in the index.

Throughput knobs (`--max-in-flight`, `evaluation.batch_size`) are excluded from the lock, so you can change concurrency and still resume.

## Secrets

`[secrets]` names are **workflow** secrets, consumed by your rollout, grader, or tool code. They resolve locally in this order: `--secrets-file`, then the process environment, then an interactive prompt. A name existing in the platform secret store does **not** satisfy them locally.

LLM provider keys resolve the same way — a `--secrets-file` entry is exported to the environment where LiteLLM reads it — but they stay on your machine: the container only ever sees the bridge's loopback URL and a per-run bearer, never the provider key. Everything that lands in the run log is redacted.

Secret values never appear in `manifest.json`, `events.jsonl`, or `logs.txt` — rollout-server output is redacted on the way into the log, and `logs.txt` and the journal are owner-only.

## Harbor (sandboxed) rollouts

Harbor rollouts work through the same command. Two things are specific to them.

**Egress.** The container needs exactly one thing: outbound access to the chat-endpoint host (on macOS the bridge's loopback URL is rewritten to `host.docker.internal`). Harbor's default network mode is `public`, where nothing is required. When a task declares `network_mode = "allowlist"`, the chat-endpoint host is added automatically — you do not list it yourself.

The two places Harbor takes run-specific allowlist entries resolve into independent policies, so each is decided on its own: the environment baseline gets the host when `[environment]` is `allowlist`, and the agent phase gets it when the resolved agent policy is `allowlist` for every step. A phase that declares `no-network` is never modified. That matters more than it sounds: Harbor turns *any* non-public policy handed extra hosts into an allowlist, so injecting blindly would put an intentionally offline agent — or, through the inherited baseline, an offline verifier — back on the network.

> **Docker and `allowlist`.** Harbor's Docker provider enforces allowlists with an egress-control sidecar built on nftables. On Linux it is always available. On macOS it depends on the Docker Desktop VM kernel exposing `CONFIG_NFT_FIB_INET`, which Harbor checks once with a probe container; when the symbol is missing, egress control is disabled and the task is rejected with `network_mode='allowlist' is not supported by EnvironmentType.DOCKER environment`. Docker Desktop builds vary, so treat that message on macOS as "this host's kernel lacks the feature", not "the provider cannot do it" — the same task runs under `allowlist` on Linux.

**Per-trial logs.** Harbor writes its own `trial.log`, `agent/`, and `verifier/` output per trial, and removes the trial directory once a successful trial's artifacts are relocated. Those files are copied to `rollout_trials/<rollout-id>/logs/` first, so they outlive cleanup. They are copied *after* credential scrubbing, and they stay out of `artifacts/` because that tree is the one the platform enumerates and renders.

## Reading a failed run

Failed rows are printed at the end with the directory that explains them:

```text
row 0 (source 3) run 0: error_type=grader_timeout -> .osmosis/evals/my-run/rollout_trials/a1b2…/
```

The source annotation appears only when `--rows` made the two indices differ. `logs.txt` carries the combined supervisor and rollout-server output for the whole run; `rollout_trials/<id>/diagnostics.json` and, for Harbor, `rollout_trials/<id>/logs/` carry the per-rollout detail.

If a process-wide fault stops the run — the rollout server dies, the model provider rejects the key — dispatch halts and the remaining work items are left **pending**, not marked failed. Fix the cause and re-run the same `--name`; only the work that never ran is dispatched.

## Known limitations

- Harbor containers reach the bridge via `host.docker.internal`, which the loopback rewrite applies on macOS only. On Linux Docker the container cannot reach a loopback-bound bridge yet, so Harbor rollouts that call the model are macOS-only for now; `LocalBackend` rollouts work everywhere.
- A `kill -9` of the supervisor can leave one rollout server running; stop it with `pkill -f <entrypoint>`. Ctrl-C and normal exits clean up after themselves.
- An in-sandbox LLM-judge grader works only if your Harbor environment already provides its own credentials: `GraderContext` carries no LLM endpoint.
- `allowlist` egress is not yet validated end-to-end; it needs Linux Docker, or a macOS host whose Docker Desktop kernel has `CONFIG_NFT_FIB_INET` (see above).
- Windows/WSL2 is out of scope and untested.
- Uploading a local run to the platform is not implemented yet.
