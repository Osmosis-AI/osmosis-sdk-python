# Rollout SDK

> The library API you implement against. Anchored to [../osmosis_ai/rollout/__init__.py](../osmosis_ai/rollout/__init__.py). For how rollouts run end to end see [architecture.md](./architecture.md); for usage and the `osmosis rollout` CLI see [docs.osmosis.ai](https://docs.osmosis.ai/cli/rollout/overview).

A rollout has two halves you provide: an `AgentWorkflow` (the agent loop) and a `Grader` (turns the trajectory into one reward). The SDK runs them behind an execution backend and the FastAPI server.

## Public surface

Everything below is re-exported from `osmosis_ai.rollout` unless noted.

| Symbol | Source | Purpose |
|--------|--------|---------|
| `AgentWorkflow` | [agent_workflow.py](../osmosis_ai/rollout/agent_workflow.py) | ABC you subclass; implement `async run(ctx)` |
| `Grader` | [grader.py](../osmosis_ai/rollout/grader.py) | ABC you subclass; implement `async grade(ctx)` |
| `AgentWorkflowContext`, `HarborAgentWorkflowContext`, `GraderContext`, `RolloutContext`, `get_rollout_context` | [context.py](../osmosis_ai/rollout/context.py) | Execution context passed to `run` / `grade` |
| `AgentWorkflowConfig`, `GraderConfig`, `ConcurrencyConfig` | [types/config.py](../osmosis_ai/rollout/types/config.py) | Pydantic config models |
| `RolloutSample`, `RolloutStatus`, `RolloutErrorCategory` | [types/sample.py](../osmosis_ai/rollout/types/sample.py) | Sample + status types |
| `create_rollout_server`, `ControllerAuth` | [server/](../osmosis_ai/rollout/server/) | FastAPI factory + bearer auth |
| `ExecutionBackend`, `LocalBackend` | [backend/](../osmosis_ai/rollout/backend/) | Execution backends |
| `OsmosisStrandsAgent`, `OsmosisRolloutModel` | [integrations/agents/strands.py](../osmosis_ai/rollout/integrations/agents/strands.py) | Strands integration |

## AgentWorkflow

```python
class AgentWorkflow[TConfig: AgentWorkflowConfig](ABC):
    def __init__(self, config: TConfig | None = None): ...
    @abstractmethod
    async def run(self, ctx: AgentWorkflowContext[TConfig]) -> Any: ...
```

[../osmosis_ai/rollout/agent_workflow.py](../osmosis_ai/rollout/agent_workflow.py)

- `run` is **async** (enforced by [validator.py](../osmosis_ai/rollout/validator.py)).
- `ctx.prompt` is the initial message list; `ctx.config` is your typed config.
- The return value is not the trajectory. The rollout's single sample is collected from the active `RolloutContext` (see [Sample](#sample)); the integrations register its source for you.

## Grader

```python
class Grader(ABC):
    def __init__(self, config: GraderConfig | None = None): ...
    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any: ...
```

[../osmosis_ai/rollout/grader.py](../osmosis_ai/rollout/grader.py)

- `ctx.sample` is the rollout's `RolloutSample`, or `None` if the workflow produced no sample.
- Attach the reward with `ctx.set_reward(reward)` — it raises `ValueError` when `ctx.sample` is `None` ([context.py](../osmosis_ai/rollout/context.py)).
- `ctx.label` carries the dataset row's label (the ground-truth string).
- `ctx.metadata` is the read-only input-side dataset row metadata.

## Contexts

[../osmosis_ai/rollout/context.py](../osmosis_ai/rollout/context.py)

- `AgentWorkflowContext` — `prompt: list[dict]`, `config`.
- `HarborAgentWorkflowContext` — adds `environment` (Harbor `BaseEnvironment`) for `environment.exec()`, `environment.upload_file()`, etc. under `HarborBackend`.
- `GraderContext` — `label`, `sample`, `metadata`, `artifacts_dir`, plus `set_reward()` for the sample's output reward.
- `RolloutContext` — ambient per-rollout context (chat completions URL, API key, rollout id). It is a context manager; the server enters it around execution. Local backends pass connection info directly; container runners read it from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` / `OSMOSIS_ROLLOUT_ID`. Fetch the current one with `get_rollout_context()`.

### Sample

The workflow does not return its sample; instead one `SampleSource` is registered on the **ambient** `RolloutContext` (fetched with `get_rollout_context()`, not the `ctx` passed to `run`) and called lazily at collection time:

```python
from osmosis_ai.rollout import get_rollout_context

rollout_ctx = get_rollout_context()              # the active RolloutContext
rollout_ctx.set_sample_source(source)             # exactly one source per rollout
sample = await rollout_ctx.get_sample()           # async -> RolloutSample | None
```

Registering a second source raises `ValueError`: one rollout is one agent execution, one sample, and one reward. `OsmosisStrandsAgent` registers its source automatically, so most workflows never call `set_sample_source` directly.

`RolloutSample` ([types/sample.py](../osmosis_ai/rollout/types/sample.py)) fields: `messages`, `label`, `reward`, `remove_sample`, `metrics`, `extra_fields`. It has no sample id; rollout identity comes from rollout-scoped URLs. The internal `trajectory_messages` field holds the normalized copy used for persistence.

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

The server saves every finished rollout as an [ATIF](https://www.harborframework.com/docs/agents/trajectory-format) trajectory document (Harbor's Agent Trajectory Interchange Format). Saving is a server-level concern — it observes the `ExecutionResult` at the backend boundary and works identically with any backend. It is always on and needs no configuration: documents are written to the same platform-managed directory as file artifacts (`~/.osmosis/<rollout_id>/`), which the platform persists to durable storage.

Layout per rollout, keyed by `rollout_id` (callers that need position semantics — e.g. an eval run's row/run index — keep them in their own index and join on the rollout id, which is also echoed in `extra.osmosis`):

```
~/.osmosis/<rollout_id>/
├── trajectory.json          # the rollout's single ATIF document
└── artifacts/...            # file artifacts (see above)
```

Each document carries a normalized, controller-compatible transcript as ATIF steps (tool calls fold into agent-step observations) and namespaces platform context under `extra.osmosis`: `rollout_id`, `label`, `reward`, sample `metrics`/`extra_fields`, and the request's `metadata`/`extra_fields` (the natural channel for run identity such as an eval run id).

Built-in sample sources keep their framework-native `RolloutSample.messages` for graders and callbacks and prepare a separate `trajectory_messages` copy through the framework converter used for OpenAI-compatible `/chat/completions` traffic. `trajectory_messages` is SDK-internal: it crosses backend boundaries for persistence but is omitted from grader callbacks. This is not an exact wire replay because call-specific conversion arguments and separately supplied system instructions are not retained. Framework-native items omitted by the framework converter are outside the persisted transcript contract. Custom sample sources whose native history is already OpenAI chat-completions-shaped get the same behavior by default. A source with another native shape sets `RolloutSample.trajectory_messages` itself from `get_sample` (an explicit `None` marks conversion as unavailable and skips trajectory persistence for that sample). Like artifacts, conversion and saving are best-effort: failures are logged and never affect rewards, callbacks, or rollout status.

### Per-call metrics (model, tokens, cost, logprobs)

ATIF has first-class slots for LLM operational data (`Step.metrics`, `Step.model_name`, `final_metrics`), but agent frameworks drop response metadata (usage, model, logprobs) when they append to the conversation. Neither the native `messages` nor the normalized `trajectory_messages` can recover that data. Two opt-in channels feed those slots; both are best-effort and change nothing when unused:

1. **Controller report (callback ack)** — the controller may attach a `trajectory` object to the JSON body of its completion/grader callback response ([report.py](../osmosis_ai/rollout/trajectory/report.py) defines the shape). Its LLM bridge serves every completion, so it is the party that has per-call usage.
   - **When to report**: snapshot the agent-phase calls into the **completion** ack, before resolving any internal future that triggers controller-side cleanup. Omit `trajectory` from the grader ack — an ack without a report keeps the earlier one, and grader-phase LLM calls (an LLM judge) would skew call counts and totals. A grader ack that does carry a report replaces the completion one entirely (no merge).
   - **Attribution**: `llm_call_metrics` map onto agent steps in dispatch order only when the counts match exactly; on a mismatch they are preserved under `extra.osmosis.unmatched_llm_call_metrics` instead of being mis-attributed, and totals still aggregate into `final_metrics`. The SDK always fills `final_metrics.total_steps` from the emitted ATIF steps.
   - **Report entry key**: `TrajectoryReport` retains a `samples` map for controller compatibility, but the SDK sample has no id. When the map contains exactly one entry, the SDK applies it to the rollout's sample regardless of the key. Multiple entries cannot be attributed and are preserved under `extra.osmosis.unmatched_sample_reports`. The SDK integrations do not stamp `x-rollout-id` or `x-sample-id` headers.

```jsonc
// response body of POST <completion_callback_url> or <grader_callback_url>
{
  "status": "ok",
  "trajectory": {
    "model_name": "openai/gpt-5-mini",
    "samples": {
      "<opaque-key>": {
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

2. **Inline message metadata** — custom workflows that manage their own message list can copy `response.usage` / `response.model` onto the assistant message (top-level `usage`/`model` keys, or the `extra.response` shape harbor's converters read). Both chat-completions (`prompt_tokens`) and Responses API (`input_tokens`) field names are accepted, and `created_at`/`timestamp` fields become `Step.timestamp`. The controller report overrides inline metadata when both are present.

## Configs

[../osmosis_ai/rollout/types/config.py](../osmosis_ai/rollout/types/config.py)

```python
class ConcurrencyConfig(BaseModel):
    max_concurrent: int | None = None   # ge=1; None = backend default / no limit

class AgentWorkflowConfig(BaseConfig):   # also GraderConfig
    name: str
    description: str | None = None
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
```

- `BaseConfig` sets `extra="allow"` and `validate_assignment=True`, so you can add your own fields (model paths, tool flags) and read them off `self.config` in `run` / `grade`.
- `name` becomes the resolved agent name (1–256 chars; see `validate_backend`).
- `concurrency.max_concurrent` caps in-flight executions — raise/lower it to avoid saturating an MCP-based rollout server (see [troubleshooting.md](./troubleshooting.md)).

## Server and backends

`LocalBackend.__init__` is keyword-only and takes `workflow` / `grader` (a class or a dotted import string), plus optional `workflow_config` / `grader_config` ([backend/local/backend.py](../osmosis_ai/rollout/backend/local/backend.py)):

```python
from osmosis_ai.rollout import create_rollout_server, LocalBackend

backend = LocalBackend(
    workflow=MyWorkflow, workflow_config=MyConfig(name="my-rollout"),
    grader=MyGrader, grader_config=GraderConfig(name="my-grader"),
)
app = create_rollout_server(backend=backend)   # FastAPI: POST /rollout, GET /health
```

- `create_rollout_server` ([server/app.py](../osmosis_ai/rollout/server/app.py)) wires the protocol: it runs the backend in a background task and posts the completion + grader callbacks.
- The controller supplies rollout-scoped `chat_completions_url`, `completion_callback_url`, and `grader_callback_url` values. Routing identity lives in those URLs; `rollout_id` in request and callback bodies is optional correlation metadata, and integrations do not add per-call rollout/sample routing headers.
- `ControllerAuth` ([server/auth.py](../osmosis_ai/rollout/server/auth.py)) supplies the bearer headers for callbacks.
- `ExecutionBackend` ([backend/base.py](../osmosis_ai/rollout/backend/base.py)) is the ABC; pick one:
  - `LocalBackend` ([backend/local/](../osmosis_ai/rollout/backend/local/)) — runs workflow + grader in-process. Re-exported from `osmosis_ai.rollout`. Used by the scaffold and eval.
  - `HarborBackend` ([backend/harbor/backend.py](../osmosis_ai/rollout/backend/harbor/backend.py)) — runs the agent inside a Harbor container; pairs with `HarborAgentWorkflowContext`. It is **not** re-exported (import `from osmosis_ai.rollout.backend.harbor.backend import HarborBackend`) and requires the external `harbor` dependency.
  - `NativeHarborBackend` ([backend/native_harbor/backend.py](../osmosis_ai/rollout/backend/native_harbor/backend.py)) — turns one self-contained Harbor task into one rollout, one sample, and one verifier reward. Import it from `osmosis_ai.rollout.backend.native_harbor`; see [native-harbor-backend.md](./native-harbor-backend.md).

### Running a server

There is no `osmosis rollout serve` command. Scaffold a server with `osmosis rollout init <name>`, which writes `rollouts/<name>/main.py` wiring `LocalBackend` + `create_rollout_server` + `uvicorn` ([../osmosis_ai/templates/_scaffolds/rollout/main.py.tpl](../osmosis_ai/templates/_scaffolds/rollout/main.py.tpl)), then run it with `python main.py` (it listens on `_OSMOSIS_ROLLOUT_PORT`, default 8000).

## Integrations

[../osmosis_ai/rollout/integrations/agents/](../osmosis_ai/rollout/integrations/agents/)

- **Strands** — `OsmosisStrandsAgent` / `OsmosisRolloutModel` are re-exported from `osmosis_ai.rollout`. `OsmosisStrandsAgent` is a drop-in for `strands.Agent`: it swaps in the rollout model from the active `RolloutContext` and auto-registers a sample source.
- **OpenAI Agents** — `OsmosisAgent` is a drop-in for `agents.Agent`, but it is **only** importable from the submodule, not re-exported by `osmosis_ai.rollout` or the integrations `__init__`:

  ```python
  from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent
  ```

## Minimal example

```python
from typing import Any
from osmosis_ai.rollout import (
    AgentWorkflow, AgentWorkflowConfig, AgentWorkflowContext,
    Grader, GraderConfig, GraderContext, OsmosisStrandsAgent, OsmosisRolloutModel,
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
            raise ValueError("workflow produced no sample")
        reward = 1.0 if str(ctx.label) in str(ctx.sample.messages[-1]) else 0.0
        ctx.set_reward(reward)
```

For complete, runnable rollouts (local Strands, local OpenAI Agents, Harbor) see the [Osmosis-AI/workspace-template](https://github.com/Osmosis-AI/workspace-template) `rollouts/` directory.

## See also

- [architecture.md](./architecture.md) — protocol + execution model
- [troubleshooting.md](./troubleshooting.md) — timeouts, event-loop blocking, concurrency
