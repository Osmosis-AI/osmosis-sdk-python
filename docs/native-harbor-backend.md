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
from harbor.models.trial.config import AgentConfig

from osmosis_ai.rollout.backend.native_harbor import NativeHarborBackend
from osmosis_ai.rollout.server import create_rollout_server


backend = NativeHarborBackend(
    agent=AgentConfig(
        name="terminus-2",                 # the default; training-safe binding
        model_name="openai/osmosis-rollout",
        override_setup_timeout_sec=300,    # setup/install, not the agent run
    ),
    max_concurrent=4,                      # running Trials
    max_queue_depth=4,                     # accepted rollouts waiting for a slot
)
app = create_rollout_server(backend=backend)   # required module-level ASGI app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")))


if __name__ == "__main__":
    main()
```

The model endpoint and key are **not** configured here — they arrive per rollout from the ambient `RolloutContext` (see [Model endpoint injection](#model-endpoint-injection)). To exercise the server locally, set the same env vars a training controller would: `OSMOSIS_CHAT_COMPLETIONS_URL`, `OSMOSIS_API_KEY`, `OSMOSIS_ROLLOUT_ID` ([context.py](../osmosis_ai/rollout/context.py)).

The plain Harbor `Trial` path used here does not invoke Harbor's telemetry
reporting call sites. Managed images should nevertheless set
`HARBOR_TELEMETRY=0` as a belt-and-suspenders opt-out.

Codex, OpenCode, and Claude Code are eval-only bindings that need protocol
translation. For any of them, also pass `gateway_base_url` as the fixed, externally reachable
origin of this same rollout server. `create_rollout_server` then mounts the
translation routes automatically. The URL must not include a path; the backend
adds the binding-specific `/v1` prefix where needed.

### Supplying full Harbor configuration

The Quickstart leaves `environment` and `verifier` at their defaults. The
canonical constructor accepts Harbor's complete `AgentConfig`,
`EnvironmentConfig`, and `VerifierConfig`, including skills, MCP servers, log
filters, network settings, mounts, resource controls, and custom verifier fields:

```python
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig

backend = NativeHarborBackend(
    agent=AgentConfig(
        name="terminus-2",
        model_name="openai/osmosis-rollout",
        skills=["org/my-skill@sha256:..."],
        include_logs=["*.json"],
        extra_allowed_hosts=["models.example.com"],
    ),
    environment=EnvironmentConfig(
        type=EnvironmentType.DAYTONA,
        override_cpus=4,
    ),
    verifier=VerifierConfig(
        max_timeout_sec=300,
        include_logs=["reward.json"],
    ),
    max_concurrent=8,
)
```

Each Trial is one environment instance, so keep `max_concurrent` aligned with the host/remote capacity (see [Concurrency and trial directories](#concurrency-and-trial-directories)). For managed SkyPilot runs, an explicit `environment.kwargs["context_name"]` wins; otherwise the backend reads `HARBOR_SKYPILOT_CONTEXT`. On macOS local Docker, controller URLs using `localhost` or `127.0.0.1` are rewritten to `host.docker.internal` so the agent container can reach them.

### Prewarm setup before the server becomes ready

`NativeHarborBackend.prewarm()` runs one Harbor `TrialConfig(install_only=True)`
per supplied `TaskConfig`. The convenience `prewarm_lifespan()` turns that into
a startup gate for the same FastAPI server:

```python
from pathlib import Path

from harbor.models.trial.config import TaskConfig


prewarm_tasks = [
    TaskConfig(path=Path("./tasks/foo")),
    TaskConfig(name="org/task", ref="sha256:9f2c..."),
    TaskConfig(
        git_url="https://example.com/org/tasks.git",
        path=Path("tasks/bar"),
        git_commit_id="abc123...",
    ),
]

app = create_rollout_server(
    backend=backend,
    lifespan=backend.prewarm_lifespan(prewarm_tasks),
)
```

The lifespan finishes prewarming before FastAPI accepts health checks or rollout
requests. For an existing custom lifespan, call `await backend.prewarm(tasks)`
before yielding. The backend clones the task list and its full
agent/environment/verifier templates, assigns unique `native-prewarm-*` trial
names, and uses the same `max_concurrent` queue with Harbor retries pinned to
zero. It needs no `RolloutContext`, controller URL/key, or callbacks. Harbor
resolves the task, starts and health-checks the environment, uploads configured
skills, and runs agent setup/install; it skips the agent run and verification,
so it makes no model call and produces no reward.

Every configured task is attempted. Raised setup exceptions and Harbor
`result.exception_info` failures are reported together, and any failure aborts
server startup. The aggregate names each task and exception type, but keeps raw
setup output out of startup logs because it may contain configured credentials.
When Harbor got far enough to create a failed trial directory, the aggregate
points to it for details; earlier resolution failures explicitly say that no
directory was created. Successful prewarm trial directories follow
`cleanup_successful_trials`; failed directories that exist are retained for
inspection.
Use immutable package digests and git commits in this startup list just as in
dataset metadata.

This list is a one-shot startup preparation plan, **not** a rollout-server work
list: it does not sample, retry, grade, or schedule controller rollouts. Miles or
eval still owns all real work. Harbor describes `install_only` as a fast setup
compatibility check, and durable cache benefit depends on the environment
provider and its lifecycle configuration. In particular, default local Docker
uses `EnvironmentConfig.delete=True`; finalization removes the container, local
image, and volumes, so an installed CLI is not guaranteed to survive into a
later Trial. Prewarming can prime only the task/package/image caches that the
chosen provider actually retains. It also does not run the real Miles/eval
closed loop and makes no new training-safety or E2E claim.

## Dataset contract

Each row points at a Harbor task through a first-class `metadata` key. The dataset schema and validator are unchanged ([datasets.md](./datasets.md)); `system_prompt` / `user_prompt` stay required by the validator but are ignored at execution time. `resolve_task` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) accepts three forms of `metadata["harbor_task"]`:

```jsonc
// Local path — task directory shipped with the rollout server (recommended for v1).
// Triggered when harbor_task starts with "./", "/", or "~".
{ "system_prompt": "", "user_prompt": "", "metadata": { "harbor_task": "./tasks/foo" } }

// Package — "org/name[@ref]" (must contain a "/"); resolved via Harbor's registry + cache.
{ "system_prompt": "", "user_prompt": "", "metadata": { "harbor_task": "org/name@sha256:9f2c..." } }

// Git — set git_url; harbor_task can be any non-path marker (e.g. "git").
{ "system_prompt": "", "user_prompt": "",
  "metadata": { "harbor_task": "git", "git_url": "https://…", "task_path": "tasks/foo", "git_commit_id": "sha…" } }
```

`metadata["harbor_task"]` is **required** — a missing value raises `ValueError`. Keep its shape consistent across all rows (the dataset validator gates on a uniform `metadata` shape). Resolution, download, and the `~/.cache/harbor` content-hash cache are all handled by Harbor's `Trial.create()`; the backend never writes a loader.

Pin network-resolved tasks so every rollout in a long run executes the same
bytes. Package references without an `@ref`, or with explicit `@latest`, still
resolve as before but log a warning; use the package's immutable `sha256:`
digest. Git tasks without a non-blank `metadata["git_commit_id"]` likewise log
an actionable warning; set it to the desired commit SHA. These are warnings,
not validation errors, and warning logs never include the git URL.

`metadata["harbor_model"]` (optional) overrides the backend's `model_name` for that single row. The selected binding still validates the provider prefix: Chat Completions bindings require `openai/...` so a row cannot silently switch the agent to a different wire protocol.

> **v1 recommendation: local-path tasks.** Ship the task directories with the rollout server (baked into the image or mounted). It is offline, needs no registry auth, and matches the "everything lives on the rollout server" model. Package/git forms work but need network access (and, for packages, Harbor credentials) on the rollout host.

Dataset construction has a few controller-owned constraints:

- Platform datasets require `system_prompt` and `user_prompt` columns and at
  least four rows, even though Native ignores the prompt columns at execution.
- Eval samples 10% of rows with seed 42 when no limit is set. Set an explicit
  limit that covers the whole task set when full coverage matters.
- For training, one dataset row becomes one GRPO group (eight rollouts per row
  by default).
- As an acceptance check, run eval with the `oracle` binding; every valid task
  should receive reward `1.0`.

## Constructor reference

All arguments are keyword-only ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)).

| Argument | Default | Purpose |
|----------|---------|---------|
| `agent` | `AgentConfig(name="terminus-2")` | Complete Harbor agent configuration. Its name/import path selects the validated binding; all preserved fields are cloned per rollout. |
| `environment` | `EnvironmentConfig()` | Complete Harbor environment configuration (Docker, Daytona, SkyPilot, or a custom import). |
| `verifier` | `VerifierConfig()` | Complete Harbor verifier configuration. Native always enables it because it is the reward source. |
| `agent_setup_timeout_sec` | `None` | Compatibility overlay for `AgentConfig.override_setup_timeout_sec`; prefer setting the field on `agent`. |
| `binding` | agent name | Validated wire/identity binding. Import-path agents must select `custom-chat-completions` explicitly. |
| `allow_unverified_agent` | `False` | Explicit eval-only opt-in for bindings that have not passed the real-infrastructure E2E checklist. |
| `gateway_base_url` | `None` | Fixed HTTP(S) origin of this rollout server's translation gateway. Required by the `codex`, `opencode`, and `claude-code` bindings; `create_rollout_server` mounts `/v1/responses` and `/v1/messages` on the same app. |
| `model_name` | `agent.model_name`, else `"openai/osmosis-rollout"` | Compatibility/default overlay. Per-row `metadata["harbor_model"]` wins, subject to the binding's provider restriction. |
| `reward_key` | `"reward"` | Which named verifier channel becomes the scalar reward (see [Reward mapping](#reward-mapping)). |
| `trials_dir` | `Path("native_trials")` | Where Harbor writes trial directories. |
| `task_resolver` | `resolve_task` | Override `ExecutionRequest -> TaskConfig` resolution. |
| `max_concurrent` | `8` | In-flight Trial cap (`>= 1`). Each Trial is often a container, so this bounds host load. |
| `max_queue_depth` | `max_concurrent` | Maximum accepted rollouts waiting beyond the running cap (`>= 0`). Set `0` to reject whenever all Trial slots are occupied. |
| `cleanup_successful_trials` | `True` | Delete a successful trial only after its ATIF has been validated and Harbor-collected artifacts have been copied out. |
| `agent_name`, `agent_import_path`, `agent_kwargs`, `agent_env`, `environment_config` | `None` | Compatibility shims for the original reduced surface. They cannot be mixed with the corresponding canonical object. |

### Configuration ownership and cloning

The constructor deep-clones all three Harbor objects immediately and again for
every rollout. Harbor may resolve agent skills in place, so no rollout can mutate
the caller's objects or another rollout's nested dictionaries, lists, env, skills,
or MCP definitions. Cloning uses Pydantic's `model_copy(deep=True)` rather than a
serialization round trip; Harbor's serializers intentionally redact or templatize
sensitive environment values.

| Configuration field | Native policy |
|---|---|
| `AgentConfig.name` / `import_path` | Preserve; they select the binding. Setting both is rejected. |
| `model_name` | Overlay: dataset row `harbor_model` > explicit constructor `model_name` > `agent.model_name` > SDK default. |
| `kwargs` | Preserve, then overlay binding-owned `api_base`, `llm_kwargs.api_key`, `extra_body.stream=False`, OpenCode `provider.openai.options.baseURL`, and pinned CLI version where applicable. |
| `env` | Preserve non-identity variables. User-supplied binding identity/auth-selection keys are rejected rather than silently overwritten. Direct bindings receive the controller endpoint/key; translated bindings receive the gateway origin plus an opaque, short-lived route token. |
| `skills`, `mcp_servers`, `include_logs`, `exclude_logs`, `extra_allowed_hosts` | Preserve. |
| `override_setup_timeout_sec` | Preserve unless `agent_setup_timeout_sec` explicitly overlays it. |
| `override_timeout_sec` | Preserve unless the rollout request supplies `agent_timeout_sec`. |
| `max_timeout_sec` | Preserve as the user's safety cap. With Native-owned timeout multipliers at `1.0`, it caps the request overlay. |
| `n_concurrent`, `concurrency_group`, `resume_trajectory=True` | Reject at construction; they conflict with the backend's queue/single-session ownership. |
| All `EnvironmentConfig` fields | Preserve. Managed SkyPilot fills `kwargs.context_name` only when the user left it unset. |
| `VerifierConfig.disable` | SDK-owned: `False` for real rollouts because the verifier produces the reward; Harbor sets it to `True` on setup-only prewarm Trials. |
| `VerifierConfig.override_timeout_sec` | Preserve unless the rollout request supplies `grader_timeout_sec`. |
| Other `VerifierConfig` fields, including `max_timeout_sec` | Preserve. |
| Trial-level task, name, directory, job/source/install flags, and timeout multipliers | SDK-owned; they are not constructor fields. |

## Agents

Agent support is binding-specific. A binding records the wire protocol, identity
channel, eval/training status, and an exact CLI version for installed agents.
Unknown built-ins fail at construction instead of inheriting generic `OPENAI_*`
wiring. Translated bindings require `gateway_base_url`; protocols without an
implemented binding still fail at construction with the missing capability
named in the error.

| Binding | Protocol / identity | Eval | Train | Status |
|---|---|---:|---:|---|
| `terminus-2` | Chat Completions via `kwargs["api_base"]` and `kwargs["llm_kwargs"]["api_key"]` | ✓ | ✓ | Summarization is off by default. |
| `oracle` | No model endpoint | ✓ | ✗ | Emits a construction warning; use it to validate datasets and verifiers. |
| `opencode` | OpenAI Responses at `/v1/responses`, translated to Chat Completions; binding-owned `provider.openai.options.baseURL`, bearer route token; CLI `1.18.9` | opt-in | ✗ | Requires `allow_unverified_agent=True`, `gateway_base_url`, and an `openai/...` model; trajectory linearity still needs E2E validation. |
| `codex` | OpenAI Responses at `/v1/responses`, translated to Chat Completions; bearer route token; CLI `0.146.0` | ✓ | ✗ | Requires `gateway_base_url`; emits an eval-only warning. Real-infrastructure streaming, tools, accounting, and append-only behavior remain E2E dependencies. |
| `claude-code` | Anthropic Messages at `/v1/messages`, translated to Chat Completions; `x-api-key` route token; CLI `2.1.220` | ✓ | ✗ | Requires `gateway_base_url`; emits an eval-only warning and masks Anthropic/OAuth plus Bedrock, Vertex, and Foundry selectors. Active host Bedrock mode is rejected so it cannot bypass the gateway. The same E2E dependencies remain. |
| `custom-chat-completions` | Chat Completions kwargs for `AgentConfig.import_path` | opt-in | ✗ | Explicit binding plus `allow_unverified_agent=True`; emits an eval-only warning. |

Installed-agent versions are binding-owned and cannot be overridden through
`agent.kwargs`. This prevents Harbor's default `@latest` installs from silently
changing behavior during a long run. Binding identity is also owned: OpenCode
configuration cannot override `provider.openai.options.baseURL`, and Chat
Completions bindings reject model prefixes for other providers.

`AgentConfig.override_setup_timeout_sec` controls only Harbor's setup/install
phase for each Trial. The controller-provided `ExecutionRequest.agent_timeout_sec`
remains the separate agent **run** timeout and overlays
`AgentConfig.override_timeout_sec` for that rollout. The compatibility
`agent_setup_timeout_sec` argument can still overlay the setup value.

### Model endpoint injection

Endpoint and key come from the ambient `RolloutContext` ([context.py](../osmosis_ai/rollout/context.py)) — `chat_completions_url` and `api_key` — which the controller supplies per rollout (read from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` on a container host).

Chat Completions bindings receive that endpoint directly. For OpenCode, Codex,
and Claude Code, the backend instead registers an opaque route token for the
duration of `execute()` and gives the agent the configured gateway origin.
OpenCode and Codex send the token as bearer auth to `/v1/responses`; Claude Code
sends it as `x-api-key` to
`/v1/messages`. The same FastAPI app resolves the token, uses LiteLLM to
translate Responses or Messages into Chat Completions, replaces the gateway
credential with the real controller key, and forwards to that rollout's raw
`chat_completions_url`. If the controller supplied no key, the gateway sends an
internal non-secret sentinel because LiteLLM requires a non-empty key; that
sentinel is never used for routing. Route state is removed on success or
failure, and an expired token receives `401` without reaching an upstream.

Harbor 0.20 generates `opencode.json` from host environment state that cannot
see the rollout-scoped `AgentConfig.env`, so the backend also writes the gateway
URL to the binding-owned `provider.openai.options.baseURL` config.

The configured gateway URL is an origin only (no path, query, or fragment) and
must be reachable by the Harbor environment. Local-Docker URL rewriting applies
to the agent-facing gateway URL, not the controller URL used by the server.
Streaming SSE and tool-call translation are covered by local round-trip unit
tests; validation against real Miles/eval infrastructure, including controller
token accounting, is still required by the E2E checklist and is not claimed
here.

The backend overlays the selected binding's identity slots and rejects
user-set identity keys in `agent.env`, so configuration cannot redirect model
traffic. It also masks Codex auth-file and Claude subscription/OAuth or alternate
cloud-provider selectors; active host Bedrock mode causes Claude Code
construction or execution to fail. For
kwargs-wired agents, `extra_body.stream=False` is pinned until the controllers
support that streaming path. The `oracle` binding is the exception: it invokes
the task's reference solution and needs no model endpoint.

## Append-only trajectories (training caveat)

RL training needs a single, linear, **append-only** token trajectory. Anything that rewrites the running context mid-run — summarization, compaction, subagents — forks that trajectory and corrupts the training signal. The backend does **not** gate or police this; it only sets a safe default for the built-in default agent and otherwise stays out of the way ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)):

- **`terminus-2` (the default agent)** summarizes mid-run, so the backend defaults `enable_summarize=False` + `proactive_summarization_threshold=0` on it. These are *overridable defaults* — `agent.kwargs` wins — so set `AgentConfig(kwargs={"enable_summarize": True})` to get summarization back (e.g. for long-context eval).
- **Every other binding is not training-safe.** Eval-only bindings warn when constructed, and unverified bindings additionally require an explicit opt-in. Protocol reachability alone cannot prove that an opaque installed CLI keeps one append-only trajectory.

| Agent | Run (eval — reward only) | Train (needs a linear token trajectory) |
|---|---|---|
| `terminus-2` (summarize off by default) | ✓ | ✓ |
| `oracle` | ✓ | ✗ (no model trajectory) |
| `opencode` / custom Chat Completions | opt-in, pending E2E | ✗ |
| `codex` / `claude-code` | ✓ through the SDK gateway; real-infrastructure E2E pending | ✗ |

## Reward mapping

Harbor verifiers emit a **named-channel** dict (`dict[str, float]`, e.g. `{"reward": 1.0}`), not a scalar. `_pick_reward` ([backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) collapses it: it takes the `reward_key` channel if present, else the sole value when there is exactly one channel. If multiple channels exist and none matches `reward_key`, the reward is left unset and the sample fails grading with a logged warning — set `reward_key` to the channel you want. The reward is read from the in-memory trial-level `TrialResult`, so no `reward.json` parsing is needed. A defensive legacy step-result fallback remains internal; it does not make multi-step tasks supported.

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

ATIF availability is still an agent capability: an agent that emits no `trajectory.json` can run and be graded, but there is no native document to persist. A malformed document causes the successful trial directory to be retained for inspection instead of being deleted. Multi-step tasks are unsupported; defensive loading code may preserve unexpected step-shaped output for inspection, but that is not a supported trajectory contract.

## Artifacts

The SDK does not scan the sandbox or decide which task files are artifacts. User or task code publishes selected files to Harbor's conventional `/logs/artifacts` directory (or declares additional artifacts in the Harbor task configuration), and Harbor downloads those files into the host trial directory. The backend only copies that already-collected tree to `~/.osmosis/<rollout_id>/artifacts`, alongside `trajectory.json`, before cleanup. Native has no supported multi-step artifact contract; defensive handling of unexpected step directories exists only to preserve evidence rather than discard it.

## Concurrency and trial directories

`max_concurrent` bounds running Trials through Harbor's `TrialQueue` semaphore;
because each Trial is typically a container, `max_concurrent < 1` is rejected.
`max_queue_depth` separately bounds requests already accepted by `POST /rollout`
but waiting beyond those running slots. It defaults to `max_concurrent`, so the
default backend accepts at most 16 rollouts: 8 running and 8 queued. Once that
bound is full, `/rollout` returns HTTP 429 immediately instead of spending the
controller's agent deadline in an unbounded SDK queue. Set `max_queue_depth=0`
to admit no work beyond the current in-flight cap. The reservation is held
through callbacks and trajectory persistence, then released on success,
failure, or cancellation.
Admission accounting is process-local; run one rollout-server worker per
configured capacity, or budget each worker's limits independently.

The server's `/health` response preserves the backend fields and adds a live
capacity snapshot. Protocol fields describe this configured server instance:
Chat Completions is always reachable; Responses and Messages appear only when
the translation gateway is mounted. For example, an idle Codex eval server
configured with eight running and eight queued slots reports:

```json
{
  "status": "ok",
  "backend": "native_harbor",
  "agent": "codex",
  "binding": "codex",
  "binding_protocol": "OpenAI Responses",
  "protocol_capabilities": [
    "OpenAI Chat Completions",
    "OpenAI Responses",
    "Anthropic Messages"
  ],
  "gateway_routing": "header_token",
  "evaluation_supported": true,
  "training_supported": false,
  "max_concurrency": 8,
  "max_queue_depth": 8,
  "capacity": {
    "max_concurrent": 8,
    "max_queue_depth": 8,
    "in_flight": 0,
    "queue_depth": 0,
    "available": 16,
    "accepting": true
  }
}
```

Controllers currently ignore the additional health fields, so capacity and
protocol surfacing are forward-compatible rather than full negotiation. The
real Miles/eval capacity-mismatch measurement remains an E2E dependency.

Native is explicitly single-step and single-agent; a task with scripted Harbor
steps is unsupported rather than deferred. Harbor trial retries are hard-disabled:
each rollout id owns exactly one Trial attempt and its linear model session. A
single-step trial fires the workflow callback when verification starts and the
grader callback when the trial finishes. Successful trials are removed only
after artifact relocation and durable provisional ATIF persistence; failed
trials and successful trials whose outputs could not be safely preserved are
kept for inspection. Harbor reports in-trial failures via
`result.exception_info` rather than by raising, and the backend always fires the
grader callback even on failure so the trainer never hangs waiting on a missing
reward.

## Submit preflight

`osmosis submit` normally requires a Python `AgentWorkflow` + `Grader` and rejects a rollout that has neither. Native rollouts have neither (reward comes from the Harbor verifier), so the contract check special-cases them: when the workflow fails to load, `discover_native_backend` ([eval/common/cli.py](../osmosis_ai/eval/common/cli.py)) imports the entrypoint, reads its module-level `app`, and verifies the backend marker recorded by `create_rollout_server`; only an app actually bound to a `NativeHarborBackend` (or subclass) skips the Grader requirement ([workspace_directory_contract.py](../osmosis_ai/platform/cli/workspace_directory_contract.py)). Merely importing or constructing the backend is insufficient, and constructing the app only inside `main()` is intentionally not part of the submit contract. The deeper checks (task resolves, agent exists, verifier present) remain runtime responsibilities inside `Trial.create().run()`. A self-deployed native server that never goes through `osmosis submit` is unaffected.

## See also

- [rollout-sdk.md](./rollout-sdk.md) — `create_rollout_server`, `ExecutionBackend`, `RolloutContext`, and the `LocalBackend` / `HarborBackend` alternatives.
- [architecture.md](./architecture.md) — the controller ↔ rollout-server protocol and execution model.
- [datasets.md](./datasets.md) — the dataset row contract the `metadata` task reference rides on.
