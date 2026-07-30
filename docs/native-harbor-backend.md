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
    agent_setup_timeout_sec=300, # agent setup/install only; not the run timeout
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

`metadata["harbor_model"]` (optional) overrides the backend's `model_name` for that single row. The selected binding still validates the provider prefix: Chat Completions bindings require `openai/...` so a row cannot silently switch the agent to a different wire protocol.

> **v1 recommendation: local-path tasks.** Ship the task directories with the rollout server (baked into the image or mounted). It is offline, needs no registry auth, and matches the "everything lives on the rollout server" model. Package/git forms work but need network access (and, for packages, Harbor credentials) on the rollout host.

## Constructor reference

All arguments are keyword-only ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)).

| Argument | Default | Purpose |
|----------|---------|---------|
| `agent_name` | `"terminus-2"` (when neither agent arg is set) | Built-in Harbor agent by name. It must have a validated binding in the table below. |
| `agent_import_path` | `None` | `"module:Class"` for a user-implemented `BaseAgent`. **Mutually exclusive** with `agent_name`; requires the explicit `custom-chat-completions` binding and opt-in. Registered Harbor built-ins cannot be selected through this escape hatch. |
| `agent_kwargs` | `None` | Harbor agent constructor kwargs. Binding-owned identity and installed-CLI version fields are overlaid. |
| `agent_env` | `None` | Extra agent environment (base layer); the selected binding overlays only its own identity variables. |
| `agent_setup_timeout_sec` | `None` | Positive, finite timeout for Harbor's agent setup/install phase (`AgentConfig.override_setup_timeout_sec`). This is separate from the per-request agent run timeout. |
| `binding` | agent name | Validated wire/identity binding. Import-path agents must select `custom-chat-completions` explicitly. |
| `allow_unverified_agent` | `False` | Explicit eval-only opt-in for bindings that have not passed the real-infrastructure E2E checklist. |
| `model_name` | `"openai/osmosis-rollout"` | Model id passed to Harbor. Overridable per row via `metadata["harbor_model"]`, subject to the binding's provider restriction. |
| `reward_key` | `"reward"` | Which named verifier channel becomes the scalar reward (see [Reward mapping](#reward-mapping)). |
| `trials_dir` | `Path("native_trials")` | Where Harbor writes trial directories. |
| `task_resolver` | `resolve_task` | Override `ExecutionRequest -> TaskConfig` resolution. |
| `environment_config` | Harbor `EnvironmentConfig()` | Harbor environment selector (Docker/Daytona/SkyPilot). |
| `max_concurrent` | `8` | In-flight Trial cap (`>= 1`). Each Trial is often a container, so this bounds host load. |
| `cleanup_successful_trials` | `True` | Delete a successful trial only after its ATIF has been validated and Harbor-collected artifacts have been copied out. |

## Agents

Agent support is binding-specific. A binding records the wire protocol, identity
channel, eval/training status, and an exact CLI version for installed agents.
Unknown built-ins fail at construction instead of inheriting generic `OPENAI_*`
wiring. Unsupported protocols also fail at construction with the missing
translation named in the error.

| Binding | Protocol / identity | Eval | Train | Status |
|---|---|---:|---:|---|
| `terminus-2` | Chat Completions via `kwargs["api_base"]` and `kwargs["llm_kwargs"]["api_key"]` | ✓ | ✓ | Summarization is off by default. |
| `oracle` | No model endpoint | ✓ | ✗ | Emits a construction warning; use it to validate datasets and verifiers. |
| `opencode` | Chat Completions via `OPENAI_BASE_URL` / `OPENAI_API_KEY`; CLI `1.18.9` | opt-in | ✗ | Requires `allow_unverified_agent=True` and an `openai/...` model; Harbor base-URL behavior and trajectory linearity still need E2E validation. |
| `codex` | OpenAI Responses; CLI `0.146.0` | blocked | ✗ | The controllers expose Chat Completions, so construction fails until a Responses translation gateway exists. |
| `claude-code` | Anthropic Messages; CLI `2.1.220` | blocked | ✗ | Construction fails until a Messages translation gateway exists; it is never given generic `OPENAI_*` identity. |
| `custom-chat-completions` | Chat Completions kwargs for an `agent_import_path` | opt-in | ✗ | Explicit binding plus `allow_unverified_agent=True`; emits an eval-only warning. |

Installed-agent versions are binding-owned and cannot be overridden through
`agent_kwargs`. This prevents Harbor's default `@latest` installs from silently
changing behavior during a long run. Binding identity is also owned: OpenCode
configuration cannot override `provider.openai.options.baseURL`, and Chat
Completions bindings reject model prefixes for other providers.

`agent_setup_timeout_sec` controls only Harbor's setup/install phase for each Trial. The controller-provided `ExecutionRequest.agent_timeout_sec` remains the separate agent **run** timeout: the backend maps setup to `AgentConfig.override_setup_timeout_sec` and run to `AgentConfig.override_timeout_sec` without one replacing the other.

### Model endpoint injection

Endpoint and key come from the ambient `RolloutContext` ([context.py](../osmosis_ai/rollout/context.py)) — `chat_completions_url` and `api_key` — which the controller supplies per rollout (read from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` on a container host). The backend overwrites the selected binding's corresponding identity slots so user `agent_kwargs` / `agent_env` cannot redirect model traffic. For kwargs-wired agents, `extra_body.stream=False` is pinned until the controllers support that streaming path. The `oracle` binding is the exception: it invokes the task's reference solution and needs no model endpoint.

## Append-only trajectories (training caveat)

RL training needs a single, linear, **append-only** token trajectory. Anything that rewrites the running context mid-run — summarization, compaction, subagents — forks that trajectory and corrupts the training signal. The backend does **not** gate or police this; it only sets a safe default for the built-in default agent and otherwise stays out of the way ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)):

- **`terminus-2` (the default agent)** summarizes mid-run, so the backend defaults `enable_summarize=False` + `proactive_summarization_threshold=0` on it. These are *overridable defaults* — your `agent_kwargs` win — so pass `agent_kwargs={"enable_summarize": True}` to get summarization back (e.g. for long-context eval).
- **Every other binding is not training-safe.** Eval-only bindings warn when constructed, and unverified bindings additionally require an explicit opt-in. Protocol reachability alone cannot prove that an opaque installed CLI keeps one append-only trajectory.

| Agent | Run (eval — reward only) | Train (needs a linear token trajectory) |
|---|---|---|
| `terminus-2` (summarize off by default) | ✓ | ✓ |
| `oracle` | ✓ | ✗ (no model trajectory) |
| `opencode` / custom Chat Completions | opt-in, pending E2E | ✗ |
| `codex` / `claude-code` | blocked on protocol translation | ✗ |

## Reward mapping

Harbor verifiers emit a **named-channel** dict (`dict[str, float]`, e.g. `{"reward": 1.0}`), not a scalar. `_pick_reward` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) collapses it: it takes the `reward_key` channel if present, else the sole value when there is exactly one channel. If multiple channels exist and none matches `reward_key`, the reward is left unset and the sample fails grading with a logged warning — set `reward_key` to the channel you want. The reward is read from the in-memory `TrialResult` (trial-level verifier result, falling back to the first step result that has rewards), so no `reward.json` parsing is needed.

A Harbor `TrialResult.exception_info` is authoritative: both callbacks report
failure and the sample reward remains unset, even if a verifier emitted a numeric
reward before the trial failed in a later phase. Failed trials can never be revived
into trainable or successful eval samples by a partial reward.

The dataset row's `ground_truth` is **not** required for native tasks — the Harbor task's verifier is self-contained.

## Structured diagnostics

Native results carry an emit-only diagnostics object in callback `extra_fields`
(`RolloutCompleteRequest` and, for failures discovered after agent completion,
`GraderCompleteRequest`). Existing controllers ignore this unknown field, so it
does not change callback handling, but it makes failures attributable without
parsing Harbor log text:

```json
{
  "backend": "native_harbor",
  "phase": "verification",
  "harbor_exception_type": "VerifierTimeoutError",
  "category": "agent_error",
  "timings_sec": {
    "setup": 0.12,
    "environment_setup": 8.4,
    "agent_setup": 1.7,
    "agent": 41.2,
    "verification": 3.1,
    "trial": 54.6
  }
}
```

The backend advances phase state from Harbor's `TrialQueue` lifecycle hooks.
Possible phases are `setup`, `trial_setup`, `environment_setup`, `agent_setup`,
`agent`, `verification`, `grading`, and `cancelled`. Hook durations cover the
pre-trial intervals; when Harbor supplies its own `TimingInfo`, the exact
environment, agent-setup, agent, verifier, and total-trial durations win. Values
are non-negative seconds. Successful results use the same shape with
`harbor_exception_type` and `category` set to `null`.

On failure, the exact object sent to the callback is also written to the SDK log
and archived as `~/.osmosis/<rollout-id>/diagnostics.json`. When a trajectory
exists, the same object is additionally embedded at
`trajectory.json.extra.osmosis.result_extra_fields`. The sidecar means setup or
agent failures remain inspectable even when no valid ATIF document exists.
Grader-only failures that happen after the workflow callback (for example a
verifier timeout) cannot alter the already-sent completion callback; they still
retain the structured payload in the log and final archive.

## Native ATIF trajectories

When the Harbor agent writes `agent/trajectory.json`, the backend treats that ATIF document as authoritative. It validates the document with Harbor's trajectory schema and passes its native steps, reasoning, tool calls, observations, subagent references, and metadata directly to trajectory persistence; it does not reconstruct them through `RolloutSample.messages`. Before deleting a successful trial, the backend durably writes a validated, redacted provisional document beside the collected artifacts. After `execute()` returns, the server overwrites it with the final document that normalizes the root `session_id` / `trajectory_id` to the rollout id, records the original ids under `extra.osmosis`, attaches rollout metadata and reward, and overlays exact per-call metrics reported by the controller. If provisional persistence fails, the source trial is retained.

`agent.extra` is preserved. Before the document leaves the backend, credential-shaped leaves such as `api_key`, `authorization`, `password`, `secret`, and `token` are recursively replaced with `[REDACTED]`; other agent configuration remains intact.

ATIF availability is still an agent capability: an agent that emits no `trajectory.json` can run and be graded, but there is no native document to persist. A malformed document, or multiple independent multi-step documents that cannot be represented as one trajectory without inventing structure, causes the successful trial directory to be retained for inspection instead of being deleted.

## Artifacts

The SDK does not scan the sandbox or decide which task files are artifacts. User or task code publishes selected files to Harbor's conventional `/logs/artifacts` directory (or declares additional artifacts in the Harbor task configuration), and Harbor downloads those files into the host trial directory. The backend only copies that already-collected tree to `~/.osmosis/<rollout_id>/artifacts`, alongside `trajectory.json`, before cleanup. Multi-step Harbor artifacts are retained beneath `artifacts/steps/<step-name>/`.

## Concurrency and trial directories

`max_concurrent` bounds in-flight Trials through a `TrialQueue` semaphore; because each Trial is typically a container, leaving this unbounded would exhaust the host (so `max_concurrent < 1` is rejected). Harbor trial retries are hard-disabled: each rollout id owns exactly one Trial attempt and its linear model session. A single-step trial fires the workflow callback when verification starts and the grader callback when the trial finishes; multi-step trials defer the workflow callback to the final result. Successful trials are removed only after artifact relocation and durable provisional ATIF persistence; failed trials and successful trials whose outputs could not be safely preserved are kept for inspection. Harbor reports in-trial failures via `result.exception_info` rather than by raising, and the backend always fires the grader callback even on failure so the trainer never hangs waiting on a missing reward.

## Submit preflight

`osmosis submit` normally requires a Python `AgentWorkflow` + `Grader` and rejects a rollout that has neither. Native rollouts have neither (reward comes from the Harbor verifier), so the contract check special-cases them: when the workflow fails to load, `discover_native_backend` ([eval/common/cli.py](../osmosis_ai/eval/common/cli.py)) imports the entrypoint, reads its module-level `app`, and verifies the backend marker recorded by `create_rollout_server`; only an app actually bound to a `NativeHarborBackend` (or subclass) skips the Grader requirement ([workspace_directory_contract.py](../osmosis_ai/platform/cli/workspace_directory_contract.py)). Merely importing or constructing the backend is insufficient, and constructing the app only inside `main()` is intentionally not part of the submit contract. The deeper checks (task resolves, agent exists, verifier present) remain runtime responsibilities inside `Trial.create().run()`. A self-deployed native server that never goes through `osmosis submit` is unaffected.

## See also

- [rollout-sdk.md](./rollout-sdk.md) — `create_rollout_server`, `ExecutionBackend`, `RolloutContext`, and the `LocalBackend` / `HarborBackend` alternatives.
- [architecture.md](./architecture.md) — the controller ↔ rollout-server protocol and execution model.
- [datasets.md](./datasets.md) — the dataset row contract the `metadata` task reference rides on.
