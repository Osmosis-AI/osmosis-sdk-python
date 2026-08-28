# Eval

> Product/end-user eval-run usage (results UI, metrics, config field reference) lives at [docs.osmosis.ai](https://docs.osmosis.ai/platform/evaluation-runs). This page is the **code-anchored** contract for `osmosis eval submit`: the TOML the SDK validates locally, the SDK-vs-backend validation split, and the submit flow. `osmosis eval rubric` (offline LLM-as-judge) is covered briefly at the end.

## `osmosis eval submit` (cloud eval runs)

Reads an evaluation config TOML, validates its **structure** locally, and POSTs it to the platform over the same git-sync flow as `osmosis train submit`. The two commands share one implementation — only the literal strings, the config loader, and the API call differ.

- Command shell: [../osmosis_ai/cli/commands/eval.py](../osmosis_ai/cli/commands/eval.py) (`eval_submit`)
- Handler + submit spec: [../osmosis_ai/platform/cli/eval.py](../osmosis_ai/platform/cli/eval.py) (`submit`)
- Config loader: [../osmosis_ai/platform/cli/eval_config.py](../osmosis_ai/platform/cli/eval_config.py)
- Shared submit flow: [../osmosis_ai/platform/cli/shared_submit.py](../osmosis_ai/platform/cli/shared_submit.py) (`run_cloud_submit`)
- Shared config primitives: [../osmosis_ai/platform/cli/shared_config.py](../osmosis_ai/platform/cli/shared_config.py)

```bash
osmosis eval submit configs/eval/<name>.toml        # interactive confirmation
osmosis eval submit configs/eval/<name>.toml --yes  # skip the prompt

# Run from any directory while still submitting source from the matching repo
osmosis --workspace <name> eval submit /absolute/repo/configs/eval/<name>.toml --yes

# Supply per-run values for [secrets] names without saving them
osmosis eval submit configs/eval/<name>.toml --secrets-file .env.run
vault read -field=env secret/eval | osmosis eval submit configs/eval/<name>.toml --secrets-file -
```

Without `--workspace`, the config path must resolve under `configs/eval/` inside the current Git workspace directory. With root `--workspace <name>`, the config path must be absolute; the SDK locates its containing Git workspace and requires the named platform workspace to be connected to the same repository. After that client-side guard, the submit request uses only workspace-name scope; the platform resolves rollout source from that workspace's connected repository.

### Config contract (what the SDK validates)

```toml
[experiment]                      # required
rollout = "my_rollout"            # single-segment logical name -> rollouts/my_rollout/
entrypoint = "agent.py"           # resolves under rollouts/my_rollout/
model_path = "Qwen/Qwen3-8B"
dataset = "my-dataset"
# commit_sha = "abc1234"          # optional, 7-40 hex chars

[evaluation]                      # optional; structure-only, values owned by backend
n = 4
limit = 100

[env]                             # optional
LOG_LEVEL = "info"

[secrets]                         # required for eval (use required = [] when none)
required = ["OPENAI_API_KEY"]
```

| Section | Required | SDK enforces locally |
|---------|----------|----------------------|
| `[experiment]` | yes | `rollout`, `entrypoint`, `model_path`, `dataset` present; `commit_sha` (if set) is 7-40 hex chars; `rollout` is a single-segment name resolving under `rollouts/`; unknown keys rejected (`ExperimentSection`). |
| `[evaluation]` | no | Structure only — known keys are `limit`, `n`, `batch_size`, `pass_threshold`, `agent_workflow_timeout_s`, `grader_timeout_s`; unknown keys rejected, **values forwarded unvalidated** (`_EvaluationSection`). |
| `[advanced]` | no | Passthrough — extra keys allowed and forwarded for server-side validation (`AdvancedPassthroughSection`). |
| `[env]` | no | Names match `ENV_VAR_NAME_RE` = `^[A-Z_][A-Z0-9_]*$`, must not start with `_OSMOSIS_` (reserved), values must be strings. |
| `[secrets]` | yes (eval) | Only the `required` key (a list of strings); names match `SECRET_NAME_RE` = `^[A-Z][A-Z0-9_]*$`; unknown keys rejected. Training configs may omit the section; **eval configs must include it** — use `required = []` when no secrets are needed. |

A name cannot appear in both `[env]` and `[secrets]`.

### SDK vs backend validation

The SDK validates **structure only**; the backend owns value-level semantics. So the SDK does **not** check provider/model validity, dataset existence and naming, or evaluation parameter ranges (`n`, `limit`, `pass_threshold`, timeouts) — those errors surface from the platform at submit time, not from the SDK. This is deliberate: the SDK does not track backend schema evolution.

### Submit flow (what happens locally before the POST)

`run_cloud_submit` ([shared_submit.py](../osmosis_ai/platform/cli/shared_submit.py)) runs in order:

1. Resolve the Git/workspace-directory context and validate the workspace contract. With root `--workspace`, resolve from the absolute config path and verify that the named workspace is connected to the same repository.
2. Resolve the config path and ensure it lives under `configs/eval/`.
3. Load + validate the TOML (`load_eval_submit_config`).
4. Validate `rollout`/`entrypoint` resolve under `rollouts/<rollout>/`, then validate the rollout backend.
5. Preflight a pinned `commit_sha` (fail fast on a confirmed-bad SHA before the platform clones the repo).
6. Render the confirmation summary; if `[secrets]` are referenced, fetch workspace + personal scopes and resolve each name (see [Secret resolution](#secret-resolution)), then **fail fast** on names that are neither stored nor supplied, with an `osmosis secret set <name>` hint.
7. Confirm (skipped with `--yes`), then POST via `client.submit_evaluation_run`. A missing-secret `404` is enriched with the same add-secret hint.

### Secret resolution

Names listed in `[secrets]` do not have to be saved in the platform secret store. `resolve_run_secrets` ([secret_resolution.py](../osmosis_ai/platform/cli/secret_resolution.py)) resolves each name by first hit:

1. `--secrets-file <path>` — a dotenv file of `NAME=value` lines; `-` reads stdin, for piping from a secret manager.
2. The process environment.
3. The platform secret store — a name already saved in the workspace or personal scope is left for the platform to resolve and is never sent from the CLI.
4. An interactive `getpass` prompt, only when `OutputContext.interactive` is true (rich output with a TTY). `--json` and `--plain` never prompt, even when stdin is a TTY.

Values supplied this way are never written to disk and must be re-supplied on every run. Outside an interactive output context, every unresolved name is reported at once as `INTERACTIVE_REQUIRED`; `details.missing` lists the names and `details.flags` lists `--secrets-file`, so CI shows the whole gap rather than one name per retry.

`osmosis train submit` shares this flow exactly. `osmosis benchmark submit` uses the same flag and ordering, but its configs reference secrets from several places and only `[secrets]` names may be supplied per run — see [benchmark.md](./benchmark.md#secret-references).

The result is an `OperationResult` whose next-steps point at `osmosis eval info <name>`, `osmosis eval list`, and the platform URL.

### Companion commands

The list, info, logs, and stop commands accept root `--workspace <name>` and otherwise use the current Git workspace directory ([../osmosis_ai/platform/cli/eval.py](../osmosis_ai/platform/cli/eval.py)). Download remains local-workspace scoped because it writes into a workspace-relative run layout:

- `osmosis eval list [--all] [--limit N]` — list runs for the workspace.
- `osmosis eval info <name|id> [-o root]` — run detail, results, and metrics (writes `<root>/metrics.json` in rich mode; default `.osmosis/evals/<name>/metrics.json`).
- `osmosis eval logs <name|id> [--cursor …]` — recent run logs, oldest first.
- `osmosis eval download <name|id> [--type metrics,trajectories|artifacts|logs|all] [--rows 3,7,10-20] [-o root] [--overwrite] [--yes]` — download run outputs into the fixed run-scoped layout. The default is `metrics,trajectories`; `--type` replaces that selection. Matching local sizes resume for free, and transfers over 100 MiB require confirmation unless `--yes` is supplied.
- `osmosis eval stop <name|id> [--yes]` — stop a run.

See [docs.osmosis.ai/cli/config-files](https://docs.osmosis.ai/cli/config-files) for the full config field reference.

Migration notice: rich-mode metrics exports no longer write new files under `.osmosis/metrics/`. Existing legacy files are left untouched; new eval exports use `.osmosis/evals/<name>/metrics.json`.

## `osmosis eval rubric` (offline LLM-as-judge)

Standalone, no-workspace scoring of a candidate output against a rubric via a hosted LLM judge (LiteLLM), available with `pip install "osmosis-ai[rubric]"`. It needs no platform auth and runs no rollout. The CLI command is `osmosis eval rubric`; the same logic is importable as `evaluate_rubric`:

```python
from osmosis_ai import evaluate_rubric, RubricResult  # lazy-loaded top-level exports
```

`evaluate_rubric` is an async function returning a `RubricResult` (`score: float`, `explanation: str`) with the score clamped to `[score_min, score_max]`. For the full signature, provider inference, API-key resolution, and error types, see the engine and types directly: [../osmosis_ai/eval/rubric/engine.py](../osmosis_ai/eval/rubric/engine.py), [../osmosis_ai/eval/rubric/types.py](../osmosis_ai/eval/rubric/types.py). For the CLI flag list, see the [command reference](https://docs.osmosis.ai/cli/command-reference).

## See also

- [datasets.md](./datasets.md) — dataset row contract
- [rollout-sdk.md](./rollout-sdk.md) — the workflow + grader API graded during eval runs
- [benchmark.md](./benchmark.md) — the `osmosis benchmark submit` config contract
