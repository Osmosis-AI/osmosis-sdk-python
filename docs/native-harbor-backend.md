# Native Harbor backend

> The `NativeHarborBackend` execution backend. Anchored to [../osmosis_ai/rollout/backend/native_harbor/backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py). For the rollout protocol and the other backends see [architecture.md](./architecture.md) and [rollout-sdk.md](./rollout-sdk.md); for the dataset row contract see [datasets.md](./datasets.md).

`NativeHarborBackend` turns each rollout into one native Harbor `Trial`: it resolves a Harbor task from the dataset row, runs the task's own agent against the controller-provided model endpoint, and maps the task's own verifier reward onto the rollout's single sample. You do **not** write an `AgentWorkflow`, a `Grader`, or a `SampleSource` — the Harbor task supplies the instruction, the environment, and the reward.

It uses the SDK-pinned Harbor line (`harbor[daytona]>=0.20.0,<0.21`) and is **not** re-exported from `osmosis_ai.rollout`; import it from its subpackage:

```python
from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
```

## When to use it

| You want | Use |
|----------|-----|
| Write the agent loop + grading in Python | `LocalBackend` ([rollout-sdk.md](./rollout-sdk.md)) |
| Run your Python `AgentWorkflow` inside a Harbor container | `HarborBackend` ([rollout-sdk.md](./rollout-sdk.md)) |
| Run an existing self-contained Harbor task (instruction + environment + tests) as the rollout | **`NativeHarborBackend`** |

The clean fit is a task set like Terminal Bench, where every task is already a native Harbor task that bundles its own Docker environment and `tests/`. The rollout becomes "point Harbor at the task and read back its reward" — no glue code.

## Shape: one Trial per rollout

The agent is **fixed per backend** (chosen once at construction); only the **task** and (optionally) the **model** vary per rollout, carried on the dataset row's `metadata`. Each `execute()` builds a `TrialConfig`, runs it through a bounded [`TrialQueue`](../osmosis_ai/rollout/backend/native_harbor/backend.py), reads `result.verifier_result.rewards`, and fires the workflow + grader callbacks. A rollout produces exactly one sample and one reward.

The dataset row's `system_prompt` / `user_prompt` (the wire `prompt` / `initial_messages`) are **ignored**: what enters training is the prompt the Harbor task's agent actually sends to the model endpoint, not the row text. Rows only need to carry the task reference (see [Dataset contract](#dataset-contract)).

## Quickstart

A native rollout server is the standard `create_rollout_server(backend=...)` wiring with a `NativeHarborBackend` instance. The resulting FastAPI app must be exposed as the module-level name `app`; `osmosis submit` imports that app to verify the actual backend binding without executing `main()`:

```python
import os

import uvicorn

from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


backend = NativeHarborBackend(
    agent_name="terminus-2",   # the default; in-process, training-safe
    max_concurrent=4,          # one Harbor Trial (often a container) per rollout
)
app = create_rollout_server(backend=backend)   # required module-level ASGI app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")))


if __name__ == "__main__":
    main()
```

The model endpoint and key are **not** configured here — they arrive per rollout from the ambient `RolloutContext` (see [Model endpoint injection](#model-endpoint-injection)). To exercise the server locally, set the same env vars a training controller would: `OSMOSIS_CHAT_COMPLETIONS_URL`, `OSMOSIS_API_KEY`, `OSMOSIS_ROLLOUT_ID` ([context.py](../osmosis_ai/rollout/context.py)).

### Selecting the environment (Docker / Daytona / SkyPilot)

The Quickstart leaves `environment_config` at its default. Harbor decides where each Trial runs — local Docker, a remote Daytona sandbox, etc. — through `EnvironmentConfig`; pass it explicitly to pick one:

```python
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import EnvironmentConfig as HarborEnvironmentConfig

backend = NativeHarborBackend(
    agent_name="terminus-2",                                              # constructor arg is agent_name
    environment_config=HarborEnvironmentConfig(type=EnvironmentType.DAYTONA),
    max_concurrent=8,
)
```

Each Trial is one environment instance, so keep `max_concurrent` aligned with the host/remote capacity (see [Concurrency and trial directories](#concurrency-and-trial-directories)). For managed SkyPilot runs, an explicit `environment_config.kwargs["context_name"]` wins; otherwise the backend reads `HARBOR_SKYPILOT_CONTEXT`. On macOS local Docker, controller URLs using `localhost` or `127.0.0.1` are rewritten to `host.docker.internal` so the agent container can reach them.

## Dataset contract

Each row points at a Harbor task through a first-class `metadata` key. The dataset schema and validator are unchanged ([datasets.md](./datasets.md)); `system_prompt` / `user_prompt` stay required by the validator but are ignored at execution time. `resolve_task` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) accepts three forms of `metadata["harbor_task"]`:

```jsonc
// Local path — task directory shipped with the rollout server (recommended for v1).
// Triggered when harbor_task starts with "./", "/", or "~".
{ "system_prompt": "", "user_prompt": "", "metadata": { "harbor_task": "./tasks/foo" } }

// Package — "org/name[@ref]" (must contain a "/"); resolved via Harbor's registry + cache.
{ "system_prompt": "", "user_prompt": "", "metadata": { "harbor_task": "org/name@latest" } }

// Git — set git_url; harbor_task can be any non-path marker (e.g. "git").
{ "system_prompt": "", "user_prompt": "",
  "metadata": { "harbor_task": "git", "git_url": "https://…", "task_path": "tasks/foo", "git_commit_id": "sha…" } }
```

`metadata["harbor_task"]` is **required** — a missing value raises `ValueError`. Keep its shape consistent across all rows (the dataset validator gates on a uniform `metadata` shape). Resolution, download, and the `~/.cache/harbor` content-hash cache are all handled by Harbor's `Trial.create()`; the backend never writes a loader.

`metadata["harbor_model"]` (optional) overrides the backend's `model_name` for that single row — useful when one dataset mixes tasks meant for different model ids.

> **v1 recommendation: local-path tasks.** Ship the task directories with the rollout server (baked into the image or mounted). It is offline, needs no registry auth, and matches the "everything lives on the rollout server" model. Package/git forms work but need network access (and, for packages, Harbor credentials) on the rollout host.

## Constructor reference

All arguments are keyword-only ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)).

| Argument | Default | Purpose |
|----------|---------|---------|
| `agent_name` | `"terminus-2"` (when neither agent arg is set) | Built-in Harbor agent by name (e.g. `terminus-2`, `codex`, `claude-code`). |
| `agent_import_path` | `None` | `"module:Class"` for a user-implemented `BaseAgent`. **Mutually exclusive** with `agent_name`. |
| `agent_kwargs` | `None` | Harbor agent constructor kwargs. In-process SDK wiring overlays identity fields; installed agents consume their declared CLI/ENV options. |
| `agent_env` | `None` | Extra env for an installed/CLI agent (base layer; SDK `OPENAI_*` overlays). |
| `model_name` | `"openai/osmosis-rollout"` | Model id passed to Harbor. Overridable per row via `metadata["harbor_model"]`. |
| `reward_key` | `"reward"` | Which named verifier channel becomes the scalar reward (see [Reward mapping](#reward-mapping)). |
| `trials_dir` | `Path("native_trials")` | Where Harbor writes trial directories. |
| `task_resolver` | `resolve_task` | Override `ExecutionRequest -> TaskConfig` resolution. |
| `environment_config` | Harbor `EnvironmentConfig()` | Harbor environment selector (Docker/Daytona/SkyPilot). |
| `max_concurrent` | `8` | In-flight Trial cap (`>= 1`). Each Trial is often a container, so this bounds host load. |
| `retry_config` | `None` | Harbor `RetryConfig` passed to the `TrialQueue`. |
| `cleanup_successful_trials` | `True` | Delete a successful trial only after its ATIF has been validated and Harbor-collected artifacts have been copied out. |

## Agents

Two agent kinds are wired differently, both **at the config layer** — the agent code is never modified.

| Agent kind | How it is selected | How endpoint/key reach it |
|------------|--------------------|---------------------------|
| In-process (e.g. `terminus-2`, or a custom `BaseAgent` via `agent_import_path`) | name or import path | `AgentConfig.kwargs["api_base"]` + `kwargs["llm_kwargs"]["api_key"]` |
| Installed / CLI (e.g. `codex`, `claude-code`) | built-in name | `AgentConfig.env["OPENAI_BASE_URL"]` + `env["OPENAI_API_KEY"]` |

Any agent addressed by `agent_import_path` (a **custom agent**) is passed through untouched — only the SDK identity wiring is overlaid on top of your `agent_kwargs` / `agent_env`. The terminus-2 summarize defaults ([Append-only trajectories](#append-only-trajectories-training-caveat)) apply to the built-in default agent only.

Installed agents also receive `agent_kwargs`, matching Harbor's `AgentConfig` contract. Harbor consumes agent-specific options declared by those agents (for example Cursor's `pricing` / `reasoning_effort`); endpoint and credential identity still come from the SDK-overlaid `agent_env` values below.

### Model endpoint injection

Endpoint and key come from the ambient `RolloutContext` ([context.py](../osmosis_ai/rollout/context.py)) — `chat_completions_url` and `api_key` — which the training controller supplies per rollout (read from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` on a container host). The backend overwrites the corresponding agent-config slot so the model identity can never be redirected by user-supplied `agent_kwargs` / `agent_env`. The same `execute()` path serves eval: point `chat_completions_url` at the model under test instead of a training session proxy.

## Append-only trajectories (training caveat)

RL training needs a single, linear, **append-only** token trajectory. Anything that rewrites the running context mid-run — summarization, compaction, subagents — forks that trajectory and corrupts the training signal. The backend does **not** gate or police this; it only sets a safe default for the built-in default agent and otherwise stays out of the way ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)):

- **`terminus-2` (the default agent)** summarizes mid-run, so the backend defaults `enable_summarize=False` + `proactive_summarization_threshold=0` on it. These are *overridable defaults* — your `agent_kwargs` win — so pass `agent_kwargs={"enable_summarize": True}` to get summarization back (e.g. for long-context eval).
- **Any other agent** — another built-in (`codex`, `claude-code`, …) or a custom `BaseAgent` via `agent_import_path` — passes through untouched. **Keeping its trajectory append-only is your responsibility.** Installed/CLI agents manage context in an opaque external process and are generally **not** training-safe, though they work fine for **eval / benchmarking**, where only the reward matters.

| Agent | Run (eval — reward only) | Train (needs a linear token trajectory) |
|---|---|---|
| in-process `terminus-2` (summarize off by default) or custom `BaseAgent` | ✓ | ✓ (you keep it append-only) |
| installed/CLI (`codex`, `claude-code`, …) | ✓ | generally ✗ |

## Reward mapping

Harbor verifiers emit a **named-channel** dict (`dict[str, float]`, e.g. `{"reward": 1.0}`), not a scalar. `_pick_reward` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) collapses it: it takes the `reward_key` channel if present, else the sole value when there is exactly one channel. If multiple channels exist and none matches `reward_key`, the reward is left unset and the sample fails grading with a logged warning — set `reward_key` to the channel you want. The reward is read from the in-memory `TrialResult` (trial-level verifier result, falling back to the first step result that has rewards), so no `reward.json` parsing is needed.

The dataset row's `ground_truth` is **not** required for native tasks — the Harbor task's verifier is self-contained.

## Native ATIF trajectories

When the Harbor agent writes `agent/trajectory.json`, the backend treats that ATIF document as authoritative. It validates the document with Harbor's trajectory schema and passes its native steps, reasoning, tool calls, observations, subagent references, and metadata directly to trajectory persistence; it does not reconstruct them through `RolloutSample.messages`. Before deleting a successful trial, the backend durably writes a validated, redacted provisional document beside the collected artifacts. After `execute()` returns, the server overwrites it with the final document that normalizes the root `session_id` / `trajectory_id` to the rollout id, records the original ids under `extra.osmosis`, attaches rollout metadata and reward, and overlays exact per-call metrics reported by the controller. If provisional persistence fails, the source trial is retained.

`agent.extra` is preserved. Before the document leaves the backend, credential-shaped leaves such as `api_key`, `authorization`, `password`, `secret`, and `token` are recursively replaced with `[REDACTED]`; other agent configuration remains intact.

ATIF availability is still an agent capability: an agent that emits no `trajectory.json` can run and be graded, but there is no native document to persist. A malformed document, or multiple independent multi-step documents that cannot be represented as one trajectory without inventing structure, causes the successful trial directory to be retained for inspection instead of being deleted.

## Artifacts

The SDK does not scan the sandbox or decide which task files are artifacts. User or task code publishes selected files to Harbor's conventional `/logs/artifacts` directory (or declares additional artifacts in the Harbor task configuration), and Harbor downloads those files into the host trial directory. The backend only copies that already-collected tree to `~/.osmosis/<rollout_id>/artifacts`, alongside `trajectory.json`, before cleanup. Multi-step Harbor artifacts are retained beneath `artifacts/steps/<step-name>/`.

## Concurrency and trial directories

`max_concurrent` bounds in-flight Trials through a `TrialQueue` semaphore; because each Trial is typically a container, leaving this unbounded would exhaust the host (so `max_concurrent < 1` is rejected). With no retries, a single-step trial fires the workflow callback when verification starts and the grader callback when the trial finishes. Multi-step verification is interleaved with agent execution, and a retried attempt may not be the final attempt, so those configurations fire the workflow callback from the final result instead. Successful trials are removed only after artifact relocation and durable provisional ATIF persistence; failed trials and successful trials whose outputs could not be safely preserved are kept for inspection. Harbor reports in-trial failures via `result.exception_info` rather than by raising, and the backend always fires the grader callback even on failure so the trainer never hangs waiting on a missing reward.

## Submit preflight

`osmosis submit` normally requires a Python `AgentWorkflow` + `Grader` and rejects a rollout that has neither. Native rollouts have neither (reward comes from the Harbor verifier), so the contract check special-cases them: when the workflow fails to load, `discover_native_backend` ([eval/common/cli.py](../osmosis_ai/eval/common/cli.py)) imports the entrypoint, reads its module-level `app`, and verifies the backend marker recorded by `create_rollout_server`; only an app actually bound to a `NativeHarborBackend` (or subclass) skips the Grader requirement ([workspace_directory_contract.py](../osmosis_ai/platform/cli/workspace_directory_contract.py)). Merely importing or constructing the backend is insufficient, and constructing the app only inside `main()` is intentionally not part of the submit contract. The deeper checks (task resolves, agent exists, verifier present) remain runtime responsibilities inside `Trial.create().run()`. A self-deployed native server that never goes through `osmosis submit` is unaffected.

## See also

- [rollout-sdk.md](./rollout-sdk.md) — `create_rollout_server`, `ExecutionBackend`, `RolloutContext`, and the `LocalBackend` / `HarborBackend` alternatives.
- [architecture.md](./architecture.md) — the controller ↔ rollout-server protocol and execution model.
- [datasets.md](./datasets.md) — the dataset row contract the `metadata` task reference rides on.
