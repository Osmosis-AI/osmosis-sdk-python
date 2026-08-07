# Rollout SDK

> The library API you implement against. Anchored to [../osmosis_ai/rollout/__init__.py](../osmosis_ai/rollout/__init__.py). For how rollouts run end to end see [architecture.md](./architecture.md); for usage and the `osmosis rollout` CLI see [docs.osmosis.ai](https://docs.osmosis.ai/cli/rollout/overview).

A rollout has two halves you provide: an `AgentWorkflow` (the agent loop) and a `Grader` (turns the trajectory into rewards). The framework-neutral core runs them behind an execution backend; install the `server` extra when you also need the FastAPI server.

## Public surface

`osmosis_ai.rollout` exports framework-neutral core only. Server, Harbor, and framework integrations have explicit import paths and installation extras.

| Symbol | Source | Purpose |
|--------|--------|---------|
| `AgentWorkflow` | [agent_workflow.py](../osmosis_ai/rollout/agent_workflow.py) | ABC you subclass; implement `async run(ctx)` |
| `Grader` | [grader.py](../osmosis_ai/rollout/grader.py) | ABC you subclass; implement `async grade(ctx)` |
| `AgentWorkflowContext`, `GraderContext`, `RolloutContext`, `SampleSource`, `get_rollout_context` | [context.py](../osmosis_ai/rollout/context.py) | Execution contexts and sample-source contract |
| `AgentWorkflowConfig`, `GraderConfig`, `ConcurrencyConfig` | [types/config.py](../osmosis_ai/rollout/types/config.py) | Pydantic config models |
| `AgentWorkflowOutput` | [types/output.py](../osmosis_ai/rollout/types/output.py) | Single-sample workflow return model |
| `RolloutSample`, `RolloutStatus`, `RolloutErrorCategory` | [types/sample.py](../osmosis_ai/rollout/types/sample.py) | Sample + status types |
| `ExecutionBackend`, `LocalBackend` | [backend/](../osmosis_ai/rollout/backend/) | Execution backends |

Optional features use these canonical modules:

| Extra | Import | Purpose |
|-------|--------|---------|
| `server` | `from osmosis_ai.rollout.server import create_rollout_server, ControllerAuth` | Generic FastAPI rollout server |
| `harbor` | `from osmosis_ai.rollout.backend.harbor import HarborBackend, TaskMode` | Harbor execution backend |
| `strands` | `from osmosis_ai.rollout.integrations.agents.strands import OsmosisStrandsAgent, OsmosisRolloutModel` | Strands integration |
| `openai-agents` | `from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent` | OpenAI Agents integration |

The `Messages` return type is available from `osmosis_ai.rollout.types`.

The `harbor` extra installs plain Harbor for an externally provided SkyPilot runtime. Daytona is retired, and do not install Harbor's `skypilot` extra. It is a host-side extra: it also carries what [../osmosis_ai/packaging.py](../osmosis_ai/packaging.py) needs to build the bundle wheel (and `uv` must be on `PATH`). Inside the task container only the framework-neutral core runs, so a bundle never installs the `harbor` extra — which is why the in-container runner and ATIF persistence must work on a bare install.

## AgentWorkflow

```python
class AgentWorkflow[TConfig: AgentWorkflowConfig](ABC):
    def __init__(self, config: TConfig | None = None): ...
    @abstractmethod
    async def run(
        self, ctx: AgentWorkflowContext[TConfig]
    ) -> AgentWorkflowOutput | Messages | None: ...
```

[../osmosis_ai/rollout/agent_workflow.py](../osmosis_ai/rollout/agent_workflow.py)

- `run` is **async** — the backend awaits it.
- `ctx.prompt` is the initial message list; `ctx.config` is your typed config.
- The return value is the primary trajectory source. Return an `AgentWorkflowOutput` (importable from `osmosis_ai.rollout`) with zero or one named message history in `samples` and optional finite numeric `metrics`; unknown top-level fields and non-finite metric values are rejected. A bare message list is wrapped as the single `"default"` sample. The `info` field is reserved and is not currently passed to graders.
- Return `None` to fall back to the sample collected on the active `RolloutContext` (see [Samples](#samples)); the integrations register sources for you.

## Grader

```python
class Grader(ABC):
    def __init__(self, config: GraderConfig | None = None): ...
    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any: ...
```

[../osmosis_ai/rollout/grader.py](../osmosis_ai/rollout/grader.py)

- `ctx.sample` is the single `RolloutSample` produced by the workflow; it is `None` when an explicit output contains no sample or when a `None` return has no registered ambient source.
- Attach its scalar reward with `ctx.set_reward(reward)`; this raises `ValueError` when `ctx.sample` is `None` ([context.py](../osmosis_ai/rollout/context.py)).
- `ctx.label` carries the dataset row's label (the ground-truth string).
- `ctx.metadata` is the read-only input-side dataset row metadata.

## Contexts

[../osmosis_ai/rollout/context.py](../osmosis_ai/rollout/context.py)

- `AgentWorkflowContext` — `prompt: list[dict]`, `config`. `HarborBackend` runs the workflow *inside* the task container, so it receives the same context and reaches the environment with ordinary process calls.
- `GraderContext` — `label`, singular `sample`, and input-side `metadata`, plus `set_reward()` for grading output.
- `RolloutContext` — ambient per-rollout context (chat completions URL, API key, rollout id). It is a context manager; the server enters it around execution. Local backends pass connection info directly; container runners read it from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` / `OSMOSIS_ROLLOUT_ID`. Fetch the current one with `get_rollout_context()`.

### Samples

A workflow may return one sample directly as `AgentWorkflowOutput` or a bare message list. When it returns `None`, a `SampleSource` registered on the **ambient** `RolloutContext` (fetched with `get_rollout_context()`, not the `ctx` passed to `run`) is called lazily at collection time:

```python
from osmosis_ai.rollout import get_rollout_context

rollout_ctx = get_rollout_context()  # the active RolloutContext
if rollout_ctx is None:
    raise RuntimeError("no active rollout context")
rollout_ctx.set_sample_source(source)  # exactly one source per rollout
sample = await rollout_ctx.get_sample()  # async -> RolloutSample | None
```

`OsmosisStrandsAgent` and `OsmosisMemorySession` register a source automatically, so most workflows never call `set_sample_source` directly.

`RolloutSample` ([types/sample.py](../osmosis_ai/rollout/types/sample.py)) fields: `messages`, `trajectory_messages`, `label`, `reward`, `remove_sample`, `metrics`, `extra_fields`.

## Artifacts

[../osmosis_ai/rollout/utils/file_artifacts.py](../osmosis_ai/rollout/utils/file_artifacts.py)

Artifacts are files produced by a rollout — logs, traces, generated outputs, screenshots, or other large/binary data that does not belong in the sample.

There is one rule: write files under `ctx.artifacts_dir` (available on both `AgentWorkflowContext` and `GraderContext`). It is `Path | None`, so check `if ctx.artifacts_dir:` before using it. In Harbor-backed rollouts it is the sandbox's `/logs/artifacts/`; `LocalBackend` provides an isolated per-rollout directory. If a file is produced elsewhere, copy it in once you've confirmed the dir exists (`if ctx.artifacts_dir: shutil.copy2(path, ctx.artifacts_dir / "name")`).

```python
import json


async def grade(self, ctx: GraderContext) -> Any:
    if ctx.artifacts_dir:
        (ctx.artifacts_dir / "trace.json").write_text(
            json.dumps({"score_reason": "matched rubric"})
        )
    ctx.set_reward(1.0)
```

After each rollout the artifacts land on the host under `~/.osmosis/<rollout_id>/artifacts/`. `LocalBackend` writes your files at that root. Harbor mirrors its collected-trial layout, so the `/logs/artifacts/` convention dir lands at `.../artifacts/logs/artifacts/<file>`, next to any paths you declare in the task's `artifacts` config.

The backend's directory setup and collection is best-effort and never affects rewards or rollout status. Writes you make in `run` or `grade` run as normal code. An unguarded write that raises will fail that workflow or grader, so always check `if ctx.artifacts_dir:` before writing to it.

## Trajectory saving

[../osmosis_ai/rollout/trajectory/](../osmosis_ai/rollout/trajectory/)

The server saves every finished rollout as an SDK-owned implementation of the ATIF trajectory schema. Saving is a server-level concern — it observes the `ExecutionResult` at the backend boundary and works identically with any backend. The generic server does not import or depend on Harbor. Saving is always on and needs no configuration: documents are written to the same platform-managed directory as file artifacts (`~/.osmosis/<rollout_id>/`), which the platform persists to durable storage.

Layout per rollout, keyed by `rollout_id` (callers that need position semantics — e.g. an eval run's row/run index — keep them in their own index and join on the rollout id, which is also echoed in `extra.osmosis`):

```
~/.osmosis/<rollout_id>/
├── trajectory.json          # the rollout's ATIF document
└── artifacts/...            # file artifacts (see above)
```

Each document carries a normalized, controller-compatible transcript as ATIF steps (tool calls fold into agent-step observations) and namespaces platform context under `extra.osmosis`: `rollout_id`, `label`, `reward`, sample `metrics`/`extra_fields`, and the request's `metadata`/`extra_fields` (the natural channel for run identity such as an eval run id).

Built-in sample sources keep their framework-native `RolloutSample.messages` for graders and callbacks and prepare a separate `trajectory_messages` copy through the framework converter used for OpenAI-compatible `/chat/completions` traffic. `trajectory_messages` is SDK-internal: it crosses backend boundaries for persistence but is omitted from grader callbacks. This is not an exact wire replay because call-specific conversion arguments and separately supplied system instructions are not retained. Framework-native items omitted by the framework converter are outside the persisted transcript contract. Custom sample sources whose native history is already OpenAI chat-completions-shaped get the same behavior by default. A source with another native shape sets `RolloutSample.trajectory_messages` itself from `get_sample` (an explicit `None` marks conversion as unavailable and skips trajectory persistence for that sample). Like artifacts, conversion and saving are best-effort: failures are logged and never affect rewards, callbacks, or rollout status.

### Per-call metrics (model, tokens, cost, logprobs)

ATIF has first-class slots for LLM operational data (`Step.metrics`, `Step.model_name`, `final_metrics`), but agent frameworks drop response metadata (usage, model, logprobs) when they append to the conversation. Neither the native `messages` nor the normalized `trajectory_messages` can recover that data. Two opt-in channels feed those slots; both are best-effort and change nothing when unused:

1. **Controller report (callback ack)** — the controller may attach a `trajectory` object to the JSON body of its completion/grader callback response ([report.py](../osmosis_ai/rollout/trajectory/report.py) defines the shape). Its LLM bridge serves every completion, so it is the party that has per-call usage.
   - **When to report**: snapshot the agent-phase calls into the **completion** ack, before resolving any internal future that triggers controller-side cleanup. Omit `trajectory` from the grader ack — an ack without a report keeps the earlier one, and grader-phase LLM calls (an LLM judge) would skew call counts and totals. A grader ack that does carry a report replaces the completion one entirely (no merge).
   - **Attribution**: `llm_call_metrics` map onto agent steps in dispatch order only when the counts match exactly; on a mismatch they are preserved under `extra.osmosis.unmatched_llm_call_metrics` instead of being mis-attributed, and totals still aggregate into `final_metrics`. The SDK always fills `final_metrics.total_steps` from the emitted ATIF steps.
   - **Sample keys**: the SDK no longer has sample ids. For a single-sample rollout, one `samples` entry is accepted regardless of its key. Multiple entries cannot be attributed; they are logged and preserved under `extra.osmosis.unmatched_sample_reports`.

```jsonc
// response body of POST <completion_callback_url> or <grader_callback_url>
{
  "status": "ok",
  "trajectory": {
    "model_name": "openai/gpt-5-mini",
    "samples": {
      "<arbitrary-key>": {
        "llm_call_metrics": [
          {"prompt_tokens": 120, "completion_tokens": 40, "cached_tokens": 0,
           "cost_usd": 0.0003, "logprobs": [-0.1], "model_name": "...",
           // exact engine tokenization (TITO); field names mirror ATIF Metrics
           "prompt_token_ids": [101, 102], "completion_token_ids": [103]}
        ],
        "final_metrics": {"total_prompt_tokens": 120}   // optional token/cost overrides; total_steps is SDK-owned
      }
    }
  }
}
```

   The server reads only the `trajectory` key off the ack body; the surrounding ack fields — `status`, `ok`, or anything else the controller returns — are accepted and ignored.

2. **Inline message metadata** — custom workflows that manage their own message list can copy `response.usage` / `response.model` onto the assistant message (top-level `usage`/`model` keys, or a compatible `extra.response` shape). Both chat-completions (`prompt_tokens`) and Responses API (`input_tokens`) field names are accepted, and `created_at`/`timestamp` fields become `Step.timestamp`. The controller report overrides inline metadata when both are present.

## Configs

[../osmosis_ai/rollout/types/config.py](../osmosis_ai/rollout/types/config.py)

```python
class ConcurrencyConfig(BaseModel):
    max_concurrent: int | None = None  # ge=1; None = backend default / no limit


class AgentWorkflowConfig(BaseConfig):  # also GraderConfig
    name: str
    description: str | None = None
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
```

- `BaseConfig` sets `extra="allow"` and `validate_assignment=True`, so you can add your own fields (model paths, tool flags) and read them off `self.config` in `run` / `grade`.
- `name` becomes the resolved agent name.
- `concurrency.max_concurrent` caps in-flight executions — raise/lower it to avoid saturating an MCP-based rollout server (see [troubleshooting.md](./troubleshooting.md)).

## Server and backends

`LocalBackend.__init__` is keyword-only and takes `workflow` / `grader` (a class or a dotted import string), plus optional `workflow_config` / `grader_config` ([backend/local/backend.py](../osmosis_ai/rollout/backend/local/backend.py)):

```python
from osmosis_ai.rollout import LocalBackend
from osmosis_ai.rollout.server import create_rollout_server

backend = LocalBackend(
    workflow=MyWorkflow,
    workflow_config=MyConfig(name="my-rollout"),
    grader=MyGrader,
    grader_config=GraderConfig(name="my-grader"),
)
app = create_rollout_server(backend=backend)  # FastAPI: POST /rollout, GET /health
```

- `create_rollout_server` ([server/app.py](../osmosis_ai/rollout/server/app.py)) is provided by the `server` extra. It wires the protocol: it runs the backend in a background task and posts the completion + grader callbacks. It has no Harbor dependency.
- `ControllerAuth` ([server/auth.py](../osmosis_ai/rollout/server/auth.py)) supplies the bearer headers for callbacks.
- `ExecutionBackend` ([backend/base.py](../osmosis_ai/rollout/backend/base.py)) is the ABC; pick one:
  - `LocalBackend` ([backend/local/](../osmosis_ai/rollout/backend/local/)) — runs workflow + grader in-process. Re-exported from `osmosis_ai.rollout`. Used by the scaffold and eval.
  - `HarborBackend` ([backend/harbor/backend.py](../osmosis_ai/rollout/backend/harbor/backend.py)) — runs the agent inside a Harbor container. It is **not** re-exported from `osmosis_ai.rollout`; import it from its canonical module (`from osmosis_ai.rollout.backend.harbor import HarborBackend`), which requires the `harbor` extra.

### Harbor backend

[../osmosis_ai/rollout/backend/harbor/](../osmosis_ai/rollout/backend/harbor/)

`agent=` picks the track. A registered native agent name (`"terminus-2"`, `"mini-swe-agent"`, `"oracle"`) runs Harbor's own agent with the rollout endpoint injected; an `AgentWorkflow` class (or `"module:Class"` path) is packaged into a wheel and installed in the task container at trial start. `grader=None` makes the task's own `tests/` the reward source; a `Grader` class is delivered as the verifier instead.

```python
from pathlib import Path

from harbor.trial.queue import TrialQueue

from osmosis_ai.rollout.backend.harbor import HarborBackend
from osmosis_ai.rollout.server import create_rollout_server

backend = HarborBackend(
    orchestrator=TrialQueue(n_concurrent=4),
    tasks_dir=Path("tasks"),
    agent=MyWorkflow,  # or a native agent name
    grader=MyGrader,  # or None to score with the task's own tests/
)
app = create_rollout_server(backend=backend, lifespan=backend.prewarm_lifespan())
```

- Tasks come from `tasks_dir` (`task_mode="template"` or `"dataset"`), or per rollout via `metadata["harbor_task"]` — a local path, a registry package `"org/name[@ref]"`, or a git checkout (`metadata["git_url"]`, ideally with a pinned `metadata["git_commit_id"]`).
- `prewarm()` builds every task image and runs agent setup before the server accepts traffic; `prewarm_lifespan()` wraps it as an ASGI lifespan.
- `max_queue_depth` bounds admission (`has_capacity()`), and `cancel_rollouts()` cancels queued or running rollouts by id, prefix, or all.

#### Migrating from the pre-v0.3 Harbor backend

v0.3 removed the original Harbor backend and gave its name to the implementation that had been called `HarborBackendV2`. The old one mounted the SDK and your source tree into the task environment and ran the workflow through an installed-agent adapter; `HarborBackend` builds a wheel from your project instead, so task images stay pure task environments.

Note that `HarborBackend` still resolves — with a different constructor. A call site passing the old keywords raises a `TypeError` naming them and pointing here; port it with this table:

| Pre-v0.3 | v0.3 |
|----------|------|
| `task_dir=` (one task) | `tasks_dir=` plus `task_mode=` (`"template"` or `"dataset"`) |
| `workflow=` | `agent=` — an `AgentWorkflow` **or** a native Harbor agent name |
| `user_code_dir=` | `code_dir=` (defaults to the agent's project dir) or a prebuilt `bundle=` |
| `grader=` or `custom_tests_dir=` (the two reward sources) | `grader=`, or `grader=None` to score with the task's own `tests/` — `custom_tests_dir=` is gone |
| `prebuild_local_image=`, `symlink_environment=` | dropped — image reuse is Harbor's, and `prewarm()` warms it |
| `HarborAgentWorkflowContext.environment` | gone: the workflow runs *inside* the container, so use ordinary process calls |

`workflow_config`, `grader_config`, `trials_dir`, `environment_config`, and `cleanup_successful_trials` carry over unchanged.

### Running a server

There is no `osmosis rollout serve` command. Scaffold a server with `osmosis rollout init <name>`, which writes `rollouts/<name>/main.py` wiring `LocalBackend` + `create_rollout_server` + `uvicorn` ([../osmosis_ai/templates/_scaffolds/rollout/main.py.tpl](../osmosis_ai/templates/_scaffolds/rollout/main.py.tpl)), then run `python rollouts/<name>/main.py` from the workspace root (it listens on `_OSMOSIS_ROLLOUT_PORT`, default 8000).

## Integrations

[../osmosis_ai/rollout/integrations/agents/](../osmosis_ai/rollout/integrations/agents/)

- **Strands** — with `pip install "osmosis-ai[strands]"`, import `OsmosisStrandsAgent` and `OsmosisRolloutModel` from `osmosis_ai.rollout.integrations.agents.strands`. `OsmosisStrandsAgent` is a drop-in for `strands.Agent`: it swaps in the rollout model from the active `RolloutContext` and auto-registers a sample source.
- **OpenAI Agents** — with `pip install "osmosis-ai[openai-agents]"`, import `OsmosisAgent` from the canonical integration module:

  ```python
  from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent
  ```

## Minimal example

```python
from typing import Any
from osmosis_ai.rollout import (
    AgentWorkflow,
    AgentWorkflowConfig,
    AgentWorkflowContext,
    Grader,
    GraderConfig,
    GraderContext,
)
from osmosis_ai.rollout.integrations.agents.strands import (
    OsmosisRolloutModel,
    OsmosisStrandsAgent,
)


class MyConfig(AgentWorkflowConfig):
    pass


class MyWorkflow(AgentWorkflow[MyConfig]):
    async def run(self, ctx: AgentWorkflowContext[MyConfig]) -> Any:
        # OsmosisRolloutModel is a placeholder; at sample time it binds to the
        # controller's model via the active RolloutContext (no model id needed here).
        agent = OsmosisStrandsAgent(name="solver", model=OsmosisRolloutModel())
        await agent.invoke_async(ctx.prompt[-1]["content"])


class MyGrader(Grader):
    async def grade(self, ctx: GraderContext) -> Any:
        if ctx.sample is None:
            raise ValueError("workflow did not produce a sample")
        reward = 1.0 if str(ctx.label) in str(ctx.sample.messages[-1]) else 0.0
        ctx.set_reward(reward)
```

For complete, runnable rollouts (local Strands, local OpenAI Agents, Harbor) see the [Osmosis-AI/workspace-template](https://github.com/Osmosis-AI/workspace-template) `rollouts/` directory.

## See also

- [architecture.md](./architecture.md) — protocol + execution model
- [troubleshooting.md](./troubleshooting.md) — timeouts, event-loop blocking, concurrency
