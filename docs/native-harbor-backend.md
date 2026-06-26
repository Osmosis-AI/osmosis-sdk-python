# Native Harbor backend

> The `NativeHarborBackend` execution backend. Anchored to [../osmosis_ai/rollout/backend/native_harbor/backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py). For the rollout protocol and the other backends see [architecture.md](./architecture.md) and [rollout-sdk.md](./rollout-sdk.md); for the dataset row contract see [datasets.md](./datasets.md).

`NativeHarborBackend` turns each rollout into one native Harbor `Trial`: it resolves a Harbor task from the dataset row, runs the task's own agent against the controller-provided model endpoint, and maps the task's own verifier reward onto the rollout's single sample. You do **not** write an `AgentWorkflow`, a `Grader`, or a `SampleSource` — the Harbor task supplies the instruction, the environment, and the reward.

It requires the external `harbor` dependency (`>=0.15`) and is **not** re-exported from `osmosis_ai.rollout`; import it from its subpackage:

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

A native rollout server is the standard `create_rollout_server(backend=...)` wiring with a `NativeHarborBackend` instance — mirror the scaffold from `osmosis rollout init`, swapping `LocalBackend` for this backend:

```python
import os

import uvicorn

from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


def main() -> None:
    backend = NativeHarborBackend(
        agent_name="terminus-2",   # the default; in-process, training-safe
        max_concurrent=4,          # one Harbor Trial (often a container) per rollout
    )
    app = create_rollout_server(backend=backend)   # FastAPI: POST /rollout, GET /health
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")))


if __name__ == "__main__":
    main()
```

The model endpoint and key are **not** configured here — they arrive per rollout from the ambient `RolloutContext` (see [Model endpoint injection](#model-endpoint-injection)). To exercise the server locally, set the same env vars a training controller would: `OSMOSIS_CHAT_COMPLETIONS_URL`, `OSMOSIS_API_KEY`, `OSMOSIS_ROLLOUT_ID` ([context.py](../osmosis_ai/rollout/context.py)).

### Selecting the environment (Docker / Daytona)

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

Each Trial is one environment instance, so keep `max_concurrent` aligned with the host/remote capacity (see [Concurrency and trial directories](#concurrency-and-trial-directories)).

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
| `agent_kwargs` | `None` | Extra constructor kwargs for an in-process agent (base layer; SDK wiring overlays — see below). |
| `agent_env` | `None` | Extra env for an installed/CLI agent (base layer; SDK `OPENAI_*` overlays). |
| `model_name` | `"openai/osmosis-rollout"` | Model id passed to Harbor. Overridable per row via `metadata["harbor_model"]`. |
| `reward_key` | `"reward"` | Which named verifier channel becomes the scalar reward (see [Reward mapping](#reward-mapping)). |
| `trials_dir` | `Path("native_trials")` | Where Harbor writes trial directories. |
| `training_safe` | `True` | Enforce append-only built-ins (see [training_safe](#training-safe)). |
| `task_resolver` | `resolve_task` | Override `ExecutionRequest -> TaskConfig` resolution. |
| `environment_config` | Harbor `EnvironmentConfig()` | Harbor environment selector (Docker/Daytona). |
| `max_concurrent` | `8` | In-flight Trial cap (`>= 1`). Each Trial is often a container, so this bounds host load. |
| `retry_config` | `None` | Harbor `RetryConfig` passed to the `TrialQueue`. |
| `cleanup_successful_trials` | `True` | Delete a successful trial's directory after reading its reward. |

## Agents

Two agent kinds are wired differently, both **at the config layer** — the agent code is never modified.

| Agent kind | How it is selected | How endpoint/key reach it |
|------------|--------------------|---------------------------|
| In-process (e.g. `terminus-2`, or a custom `BaseAgent` via `agent_import_path`) | name or import path | `AgentConfig.kwargs["api_base"]` + `kwargs["llm_kwargs"]["api_key"]` |
| Installed / CLI (e.g. `codex`, `claude-code`) | built-in name | `AgentConfig.env["OPENAI_BASE_URL"]` + `env["OPENAI_API_KEY"]` |

A **custom agent** is any `agent_import_path` whose module is outside `harbor.*`. Custom agents are passed through untouched (your `agent_kwargs` / `agent_env` are the base, SDK identity wiring overlays on top). For an in-process agent the backend also runs a preflight: kwargs the constructor cannot accept raise a clear `ValueError` instead of a cryptic `TypeError` from Harbor.

### Model endpoint injection

Endpoint and key come from the ambient `RolloutContext` ([context.py](../osmosis_ai/rollout/context.py)) — `chat_completions_url` and `api_key` — which the training controller supplies per rollout (read from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` on a container host). The backend overwrites the corresponding agent-config slot so the model identity can never be redirected by user-supplied `agent_kwargs` / `agent_env`. The same `execute()` path serves eval: point `chat_completions_url` at the model under test instead of a training session proxy.

## training_safe

RL training needs a single, linear, append-only token trajectory. Mid-run context summarization or subagents fork that trajectory and corrupt the training signal, so `training_safe=True` (the default) gates which agents are allowed and pins their append-safe knobs ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)):

- **Built-in agents** must be known-append-safe. Currently only `terminus-2` qualifies, and the backend forces `enable_summarize=False` + `proactive_summarization_threshold=0` on it. Selecting any other built-in (e.g. `codex`) under `training_safe=True` raises `ValueError`.
- **Custom agents** (`agent_import_path` outside `harbor.*`) are trusted and pass untouched — you own the append-only invariant.
- Set `training_safe=False` for **eval / benchmarking**, where only the reward matters and any CLI agent (with summarization, subagents, etc.) is fine.

| | Run (eval — reward only) | Train (needs a linear token trajectory) |
|---|---|---|
| in-process `terminus-2` (summarize off) or custom `BaseAgent` | ✓ | ✓ |
| installed/CLI (`codex`, `claude-code`, …) | ✓ (`training_safe=False`) | generally ✗ |

## Reward mapping

Harbor verifiers emit a **named-channel** dict (`dict[str, float]`, e.g. `{"reward": 1.0}`), not a scalar. `_pick_reward` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) collapses it: it takes the `reward_key` channel if present, else the sole value when there is exactly one channel. If multiple channels exist and none matches `reward_key`, the reward is left unset and the sample fails grading with a logged warning — set `reward_key` to the channel you want. The reward is read from the in-memory `TrialResult` (trial-level verifier result, falling back to the first step result that has rewards), so no `reward.json` parsing is needed.

The dataset row's `ground_truth` is **not** required for native tasks — the Harbor task's verifier is self-contained.

## Concurrency and trial directories

`max_concurrent` bounds in-flight Trials through a `TrialQueue` semaphore; because each Trial is typically a container, leaving this unbounded would exhaust the host (so `max_concurrent < 1` is rejected). Successful trials have their directory under `trials_dir` removed (`cleanup_successful_trials=True`); failed trials are kept for inspection. Harbor reports in-trial failures via `result.exception_info` rather than by raising, and the backend always fires the grader callback even on failure so the trainer never hangs waiting on a missing reward.

## Submit preflight

`osmosis submit` normally requires a Python `AgentWorkflow` + `Grader` and rejects a rollout that has neither. Native rollouts have neither (reward comes from the Harbor verifier), so the contract check special-cases them: when the workflow fails to load, `discover_native_backend` ([eval/common/cli.py](../osmosis_ai/eval/common/cli.py)) scans the entrypoint module for a `NativeHarborBackend` subclass and, if found, skips the Grader requirement ([workspace_directory_contract.py](../osmosis_ai/platform/cli/workspace_directory_contract.py)). The deeper checks (task resolves, agent exists, verifier present) cannot run statically at submit time — they are left to runtime inside `Trial.create().run()`. A self-deployed native server that never goes through `osmosis submit` is unaffected.

## See also

- [rollout-sdk.md](./rollout-sdk.md) — `create_rollout_server`, `ExecutionBackend`, `RolloutContext`, and the `LocalBackend` / `HarborBackend` alternatives.
- [architecture.md](./architecture.md) — the controller ↔ rollout-server protocol and execution model.
- [datasets.md](./datasets.md) — the dataset row contract the `metadata` task reference rides on.
