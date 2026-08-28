# `osmosis eval run` — local evaluation

Runs an evaluation on your machine against the rollout server in your own workspace, using the same TOML, the same rollout entrypoint, and the same v0.3 HTTP/callback protocol as `osmosis eval submit`. Command shell: [`osmosis_ai/cli/commands/eval.py`](../osmosis_ai/cli/commands/eval.py); business logic: [`osmosis_ai/platform/cli/eval_run.py`](../osmosis_ai/platform/cli/eval_run.py); supervisor: [`osmosis_ai/eval/local/`](../osmosis_ai/eval/local/).

## Install

```bash
pip install "osmosis-ai[eval]"
```

That extra covers the supervisor process only — the CLI, CSV/JSONL/Parquet dataset readers, the localhost callback listener, and the in-process LiteLLM bridge. The rollout server's dependencies do **not** come from here: they are resolved from `rollouts/<name>/pyproject.toml`, which the `osmosis rollout init` scaffold already declares as `osmosis-ai[server]`. A Harbor rollout adds the `harbor` extra *there*, in its own `pyproject.toml`, not next to the CLI.

Running a local eval also requires [uv](https://docs.astral.sh/uv/): it is what launches the rollout server in that environment. The `eval` extra installs it; a uv already on `PATH` works too.

## Requirements

`eval run` runs from a local Osmosis Git workspace. A platform dataset requires `osmosis auth login`; `--dataset-file <path>` without `--upload` does not load platform credentials or select a platform workspace. Adding `--upload` requires login because the completed results are imported into the platform. All LLM traffic is served by an in-process LiteLLM bridge on your machine — the same design as the hosted eval controller — so `experiment.model_path` is a LiteLLM model id (`openai/gpt-5-mini`, `anthropic/claude-sonnet-4-6`, …) and provider credentials resolve exactly as LiteLLM resolves them anywhere: from the process environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, provider `*_API_BASE` overrides, …) or from `--secrets-file`. Local OpenAI-compatible endpoints (a laptop vLLM or Ollama) work through the corresponding LiteLLM provider id and its environment variables.

The rollout entrypoint may build `LocalBackend`, `HarborBackend` with `EnvironmentType.DOCKER` (the host Docker runtime), or — with `--tunnel` or `--advertise-url` — a Harbor cloud environment (Daytona, SkyPilot, …). Cloud sandboxes have no route to your loopback interface, so the sandbox-facing model endpoint must be exposed through a tunnel; see [Cloud sandboxes](#cloud-sandboxes-tunnels). Without a tunnel flag, a cloud-environment rollout fails each work item with an error naming the fix instead of hanging.

## The rollout environment

The rollout server does not run in the CLI's environment. It is launched with `uv run --project rollouts/<name>`, in an environment resolved from that rollout's own `pyproject.toml` — the same dependency truth `osmosis eval submit` and training already use, so the rollout that runs here is the rollout that runs in the cloud. A small supervisor bootstrap adds a private ownership marker to `/health`, allowing rollout projects pinned to older SDK versions to remain compatible without weakening the check that prevents dispatching work to an unrelated local process.

The first run of a rollout resolves and installs those dependencies as its own milestone (`→ syncing rollout dependencies (<rollout>)`) and creates `rollouts/<name>/.venv`; later runs reuse it and the step is quick. Being a named stage is the point: a dependency failure is reported as a dependency failure — with the resolver's output in `logs.txt` — never as a rollout-server health timeout.

The `osmosis-ai` version inside that environment is whatever the rollout's `pyproject.toml` resolves to, which may differ from the version of the CLI you invoked. That is deliberate, and matches how cloud eval and training resolve it. To point the rollout environment at a local SDK checkout while developing, use uv's standard mechanism in `rollouts/<name>/pyproject.toml`:

```toml
[tool.uv.sources]
osmosis-ai = { path = "/path/to/osmosis-sdk-python", editable = true }
```

## Basic use

```bash
osmosis eval run configs/eval/my-rollout.toml --name my-local-run
```

The same command re-run resumes: work items with a durable terminal result are skipped, everything else runs again.

```bash
osmosis eval run configs/eval/my-rollout.toml --dataset-file data/dev.jsonl --rows 3,7,10-20
```

`--rows` selects dataset rows by their offset in the file, which is the cheapest way to smoke-test a rollout. `--dataset-file` runs a local file instead of the platform dataset named in `experiment.dataset`.

Add `--upload` to import a completed local run into the platform before `eval run` releases its run lock:

```bash
osmosis eval run configs/eval/my-rollout.toml --name my-local-run --upload
```

An existing completed run can be uploaded or resumed independently:

```bash
osmosis eval upload .osmosis/evals/my-local-run
```

## Options

| Flag | Meaning |
|---|---|
| `--name` | Stable run name. Re-running the same name resumes. Omit for a one-off run with a cloud-style `adjective-animal-number` name. |
| `--output`, `-o` | Run output root. Default `.osmosis/evals/`. |
| `--dataset-file` | Local dataset file instead of the platform dataset. |
| `--secrets-file` | Dotenv file supplying `[secrets]` values; `-` reads stdin. Values are never written to disk. |
| `--rows` | Dataset rows to run, e.g. `3,7,10-20`. Overrides `evaluation.limit`. |
| `--fresh` | Archive this name's existing results and start clean. |
| `--retry-failed` | Also re-run failed and skipped work items. Successes are kept. |
| `--max-in-flight` | Concurrent rollouts. Default: `evaluation.batch_size`, then the backend's advertised capacity, then 1. |
| `--yes`, `-y` | Skip the cost confirmation. |
| `--tunnel` | Expose the model bridge through an auto-managed tunnel so cloud sandboxes can call back. Providers: `cloudflared`. |
| `--upload` | Import completed results into the platform while the run lock is still held. |

Advanced: `--rollout-port` pins the rollout-server port, `--listener-port` pins the controller listener's local port, `--advertise-url` names a public base URL that already reaches the listener (a tunnel you run yourself; mutually exclusive with `--tunnel`), and `--verbose` streams log lines to the terminal. Before any dispatch the run makes one probe completion against the configured model, so a bad model id or key fails once, loudly, instead of failing every row.

There is deliberately **no** `[local]` or `[execution]` config section: everything that differs between a local and a cloud run is a CLI flag, so one TOML means the same thing to both commands.

## What a run prints

A plain run narrates itself: the plan table (the same one `eval submit` prints) before the cost confirmation, one line per milestone, a live progress bar while rollouts are in flight, and the metrics at the end.

```text
          Local Evaluation
╭──────────────┬───────────────────╮
│ Rollout      │ echo              │
│ Entrypoint   │ main.py           │
│ Model        │ openai/gpt-5-mini │
│ Dataset      │ multiply (cache)  │
│ Rows         │ 20 of 200         │
│ Runs Per Row │ 3                 │
│ Work Items   │ 60                │
│ Output       │ .osmosis/evals    │
╰──────────────┴───────────────────╯
→ brave-falcon-42: 60 of 60 work items pending
60 rollouts x model openai/gpt-5-mini — continue? [y/N] y
→ checking model openai/gpt-5-mini
→ syncing rollout dependencies (echo)
→ starting rollout server (main.py)
→ rollout server healthy on port 51423
→ running 60 work items, up to 8 in flight
⠹ rollouts ━━━━━━━━━━━━━━━━━━━━━━━━ 42/60 pass 90% · failed 4 0:03:11
```

The milestones are the waits worth naming: resume replay, the model preflight, the rollout dependency sync, the rollout server's startup and health, scheduling, and the grace period Ctrl-C spends unwinding cancelled rollouts. Each is also a line in `logs.txt`, written by the same call — the terminal and the log cannot drift apart.

The bar needs a terminal. Redirected output gets one printed line per completed work item instead; `--plain` and `--json` print no progress at all, and keep stdout to the result line or the envelope alone.

`--verbose` replaces the milestone lines with the full log stream — `logs.txt` line for line, rollout-server output included. The plan table, the progress display, and the results table stay.

The run ends with a results table: work items by outcome, pass rate against the threshold, reward statistics, pass@k for multi-attempt runs, tokens, duration, and the output directory. Every number in it is read straight from what `metrics.json` holds — nothing is recomputed for display, so the terminal and the file cannot disagree. Durations use the same formatter as `osmosis eval info`, so a local run and a cloud run of the same evaluation read alike. A completed run that was not uploaded with `--upload` then prints a copyable `osmosis eval upload <run-directory>` command.

## Output layout

```text
.osmosis/evals/<run-name>/
  manifest.json        # local provenance + the resolved-input lock (never uploaded)
  events.jsonl         # terminal-result journal — the resume authority
  server.json          # rollout-server ownership record; present only while one may run
  tunnel.json          # cloudflared ownership record; present only while a tunnel may run
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

A `kill -9` normally does not leak the rollout server either. The supervisor records the server's process group, together with its pid and start time, in `server.json` at spawn, and removes the record on clean shutdown. A `--tunnel` run records its cloudflared child the same way in `tunnel.json` — an orphaned quick tunnel would otherwise keep a live public URL forwarding to a local port a later run may reuse — and the next invocation reaps both records under the same ownership proof. The next invocation for the run — a plain resume, `--fresh`, even a refused resume — terminates the recorded group, but only after re-verifying that pid and start time still match: a record whose pid was recycled by an unrelated process is dropped, never signalled. Two narrow gaps remain, both resolved in favor of never signalling an unverified group: a kill that lands in the instant between the spawn and the record's write (or a record write that fails outright — the run warns and continues) leaves no record, and a group whose recorded leader died while a descendant survived can no longer be verified, so it is dropped rather than killed blind. A server that slips through either gap still needs manual cleanup.

A named run is pinned to its resolved inputs: model, dataset bytes, selected rows, `n`, timeouts, pass threshold, rollout entrypoint, and a digest of the rollout's source files. The environment the dependency sync builds is not source: `.venv/`, `uv.lock`, and a build backend's `*.egg-info/` are excluded, so resolving dependencies never changes the digest of the project that owns them. Change any of the inputs and the run refuses to resume, naming the input group that changed:

```text
Error: run 'my-local-run' was created with different resolved inputs, so resuming it would
mix versions inside one set of metrics. Changed: rollout. Restart under the same name with
--fresh (the previous results are archived, never deleted).
```

This is deliberate. The common loop — "some rows failed, edit the agent, re-run the failures" — would otherwise produce one set of metrics computed from two different versions of your code. Use `--fresh` to restart under the same name; the previous results move to `.osmosis/evals/<name>.archive-<utc-timestamp>/`.

`--retry-failed` re-runs failed and skipped items *without* a code change, which is the case where mixing versions cannot happen. Each attempt gets a fresh rollout id; the superseded attempt's directory stays for diagnosis but never appears in the index.

Throughput knobs (`--max-in-flight`, `evaluation.batch_size`) are excluded from the lock, so you can change concurrency and still resume.

## Uploading completed results

`eval upload` validates and hashes the local run under the same sibling `.locks/<run-name>.lock` used by the supervisor. Only `index.jsonl`, `progress.json`, index-referenced canonical trajectories, and safe artifacts belonging to the selected rollout ids are sent. The immutable local manifest, journal, metrics projection, summaries, logs, top-level projections, superseded attempts, per-trial logs, control files, and secret-bearing inputs are never uploaded. Local `logs.txt` is intentionally excluded because an older SDK may have written it without redacting short configured secrets.

The platform owns resumability. Each retry starts the same import from `local_run_id` plus the exact manifest digest, skips files the server already has, and finalizes once all declared hashes are present; the SDK writes no local upload-state file and does not abort a session when interrupted. A run must be complete: failed and skipped samples count as terminal, but pending work does not.

`pass_threshold` is taken from the completed `metrics.json` projection when binding the import metadata. It is not added to the shipped manifest input schema in this release, because doing so would make existing named runs refuse resume; the platform still recomputes imported results from the uploaded index and trajectories.

## Secrets

`[secrets]` names are **workflow** secrets, consumed by your rollout, grader, or tool code. They resolve locally in this order: `--secrets-file`, then the process environment, then an interactive prompt. A name existing in the platform secret store does **not** satisfy them locally.

LLM provider keys resolve the same way — a `--secrets-file` entry is exported to the environment where LiteLLM reads it — but they stay on your machine: the container only ever sees the bridge's loopback URL and a per-run bearer, never the provider key. Everything that lands in the run log is redacted.

Secret values never appear in `manifest.json` or `events.jsonl`. The current SDK redacts every non-empty configured secret value from rollout-server output on the way into `logs.txt`; the log and journal are owner-only, and local logs are not uploaded because runs produced by older SDKs cannot prove the same short-secret redaction.

## Harbor (sandboxed) rollouts

Harbor rollouts work through the same command. Two things are specific to them.

**Egress.** The container needs exactly one thing: outbound access to the chat-endpoint host (on macOS the bridge's loopback URL is rewritten to `host.docker.internal`). Harbor's default network mode is `public`, where nothing is required. When a task declares `network_mode = "allowlist"`, the chat-endpoint host is added automatically — you do not list it yourself.

The two places Harbor takes run-specific allowlist entries resolve into independent policies, so each is decided on its own: the environment baseline gets the host when `[environment]` is `allowlist`, and the agent phase gets it when the resolved agent policy is `allowlist` for every step. A phase that declares `no-network` is never modified. That matters more than it sounds: Harbor turns *any* non-public policy handed extra hosts into an allowlist, so injecting blindly would put an intentionally offline agent — or, through the inherited baseline, an offline verifier — back on the network.

> **Docker and `allowlist`.** Harbor's Docker provider enforces allowlists with an egress-control sidecar built on nftables. On Linux it is always available. On macOS it depends on the Docker Desktop VM kernel exposing `CONFIG_NFT_FIB_INET`, which Harbor checks once with a probe container; when the symbol is missing, egress control is disabled and the task is rejected with `network_mode='allowlist' is not supported by EnvironmentType.DOCKER environment`. Docker Desktop builds vary, so treat that message on macOS as "this host's kernel lacks the feature", not "the provider cannot do it" — the same task runs under `allowlist` on Linux.

**Per-trial logs.** Harbor writes its own `trial.log`, `agent/`, and `verifier/` output per trial, and removes the trial directory once a successful trial's artifacts are relocated. Those files are copied to `rollout_trials/<rollout-id>/logs/` first, so they outlive cleanup. They are copied *after* credential scrubbing, and they stay out of `artifacts/` because that tree is the one the platform enumerates and renders.

## Cloud sandboxes (tunnels)

A Harbor rollout whose `environment_config` selects a cloud environment (Daytona, SkyPilot, …) runs its sandbox off your machine, and that sandbox has no route to the bridge's loopback URL. Exactly one flow crosses that boundary — the sandbox's OpenAI-compatible chat traffic — so exposing local eval to cloud sandboxes means exposing exactly one HTTP surface: the bridge's per-rollout chat endpoint. Completion and grader callbacks come from the rollout server, a host process, and never leave your machine; neither do provider keys. The sandbox receives only the tunnel URL and the per-run bridge bearer.

```bash
osmosis eval run configs/eval/my-daytona-rollout.toml --tunnel cloudflared
```

`--tunnel cloudflared` starts a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) as a child of the run: it requires the `cloudflared` binary on `PATH` (macOS: `brew install cloudflared`), needs no account, prints the `https://<random>.trycloudflare.com` URL as a run milestone, is readiness-probed through the edge before any dispatch, and is terminated with the run. If the tunnel process dies mid-run — quick tunnels hold a single edge connection — dispatch halts with a clear error and the unfinished work items stay pending for a resume, exactly like a rollout-server death.

Tunnel mode also arms a bridge-side keepalive for non-streaming model calls: proxies enforce idle-read timeouts (cloudflared cuts a response that stays silent for ~125s), so after a 90s grace window the bridge commits `200` + `application/json` and trickles one whitespace byte every 30s until the real body is ready. Two consequences worth knowing: a provider error that occurs *after* the grace window can no longer change the status code and arrives as an OpenAI-style `{"error": ...}` body instead of a clean 502 — clients surface it as a parse/validation error — and streaming calls need no keepalive because the bridge's SSE stream already heartbeats (but see the streaming caveat below). Loopback-only runs keep exact current behavior.

Caveats of the quick-tunnel path:

- **Terms of service.** Quick tunnels are Cloudflare's free "testing and development" tier — the same mechanism the Shopify CLI spawns on every `shopify app dev`. Fine for development-scale eval runs; not a production transport.
- **Capacity.** The edge caps a tunnel at 200 in-flight requests (429 above; measured: exact and clean — excess requests are rejected, the rest are unaffected). Each rollout issues its chat calls serially, so in-flight requests ≈ `--max-in-flight`; up to ~100 adds negligible latency. There is no SLA.
- **Creation rate limit.** Cloudflare also rate-limits quick-tunnel *creation* per source IP (measured: ~20 tunnels within 30 minutes tripped HTTP 429 for 25+ minutes; tunnels already running keep serving). One run creates one tunnel, so normal use never sees it, but rapid-fire runs from one machine or CI can. The run fails fast with a rate-limit error naming the fix; wait a few minutes or use `--advertise-url`.
- **Streaming (SSE).** Cloudflare documents that quick tunnels do not support server-sent events. OpenAI-style streaming (SSE over POST) does stream in real time through them today, but that is unsupported behavior and can regress without notice. The bundled native agents default to non-streaming calls, which the keepalive covers; a rollout that depends on streaming should use `--advertise-url` with a tunnel that supports SSE (ngrok or a named Cloudflare tunnel).
- **Corporate networks.** `*.trycloudflare.com` has abuse history and is blocked by some corporate egress filters, EDR products, and filtering DNS resolvers (Tailscale MagicDNS setups included). The readiness probe runs from your machine, whose DNS says nothing about the sandbox's: when the probe gets no HTTP response at all, the run continues with a `tunnel up ... but this host cannot reach it` warning and lets the sandbox try. Only an edge that answers but keeps failing is treated as a dead tunnel. If the sandbox cannot reach it either, run your own tunnel and pass `--advertise-url`.

`--advertise-url <base-url>` is the bring-your-own escape hatch: any HTTPS reverse proxy or tunnel (ngrok, an SSH forward on a VPS, a corporate ingress) that forwards to the listener works. It requires `--listener-port` — your tunnel needs a fixed local port to forward to. Start the tunnel yourself, then pass its public base URL:

```bash
osmosis eval run configs/eval/my-daytona-rollout.toml --listener-port 8710 --advertise-url https://eval.example.com
```

Only the sandbox-facing chat endpoint uses the advertised URL. The tunnel needs to pass POST bodies and, for streaming, server-sent events; the bridge keepalive (armed for `--advertise-url` too) covers proxies with idle-read timeouts.

**What the tunnel exposes.** The listener serves no docs or OpenAPI surface, so a tunnel URL alone yields only 404s and 401s. Every route requires a bearer: the callback routes take the controller token, which never leaves your machine, and the chat route takes the per-run bridge token, which is exactly what the sandbox holds — so a compromised sandbox (or a leaked bridge token) can spend your provider credentials on arbitrary completions for the lifetime of the run, same as it already could over loopback; the tunnel widens where that token can be used from, not what it can do. Both tokens are 128-bit random values compared in constant time and rotate every run. Use `--tunnel`/`--advertise-url` for development-scale runs, not for anything you would leave running unattended against an expensive provider account.

Tasks running `allowlist` egress need no extra configuration: the tunnel host is added to the sandbox's allowlist automatically, the same way the Docker loopback host is.

## Reading a failed run

Failed rows are printed at the end with the directory that explains them:

```text
row 0 (source 3) run 0: error_type=grader_timeout -> .osmosis/evals/my-run/rollout_trials/a1b2…/
```

The source annotation appears only when `--rows` made the two indices differ. `logs.txt` carries the combined supervisor and rollout-server output for the whole run; `rollout_trials/<id>/diagnostics.json` and, for Harbor, `rollout_trials/<id>/logs/` carry the per-rollout detail.

If a process-wide fault stops the run — the rollout server dies, the model provider rejects the key — dispatch halts and the remaining work items are left **pending**, not marked failed. Fix the cause and re-run the same `--name`; only the work that never ran is dispatched.

## Known limitations

- Harbor containers reach the bridge via `host.docker.internal`, which the loopback rewrite applies on macOS only. On Linux Docker the container cannot reach a loopback-bound bridge, so a model-calling Docker rollout there is refused with the same error as a cloud environment — run it with `--tunnel cloudflared` (or `--advertise-url`) until a Linux gateway rewrite exists. `LocalBackend` rollouts work everywhere.
- An in-sandbox LLM-judge grader works only if your Harbor environment already provides its own credentials: `GraderContext` carries no LLM endpoint.
- `allowlist` egress is not yet validated end-to-end; it needs Linux Docker, or a macOS host whose Docker Desktop kernel has `CONFIG_NFT_FIB_INET` (see above).
- Windows/WSL2 is out of scope and untested.
