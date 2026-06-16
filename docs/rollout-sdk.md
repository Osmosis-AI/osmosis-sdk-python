# Rollout SDK

> The library API you implement against. Anchored to [../osmosis_ai/rollout/__init__.py](../osmosis_ai/rollout/__init__.py). For how rollouts run end to end see [architecture.md](./architecture.md); for usage and the `osmosis rollout` CLI see [docs.osmosis.ai](https://docs.osmosis.ai/cli/rollout/overview).

A rollout has two halves you provide: an `AgentWorkflow` (the agent loop) and a `Grader` (turns the trajectory into rewards). The SDK runs them behind an execution backend and the FastAPI server.

## Public surface

Everything below is re-exported from `osmosis_ai.rollout` unless noted.

| Symbol | Source | Purpose |
|--------|--------|---------|
| `AgentWorkflow` | [agent_workflow.py](../osmosis_ai/rollout/agent_workflow.py) | ABC you subclass; implement `async run(ctx)` |
| `Grader` | [grader.py](../osmosis_ai/rollout/grader.py) | ABC you subclass; implement `async grade(ctx)` |
| `AgentWorkflowContext`, `HarborAgentWorkflowContext`, `GraderContext`, `RolloutContext`, `get_rollout_context` | [context.py](../osmosis_ai/rollout/context.py) | Execution context passed to `run` / `grade` |
| `AgentWorkflowConfig`, `GraderConfig`, `ConcurrencyConfig` | [types/config.py](../osmosis_ai/rollout/types/config.py) | Pydantic config models |
| `RolloutSample`, `RolloutStatus`, `RolloutErrorCategory`, `MultiTurnMode` | [types/sample.py](../osmosis_ai/rollout/types/sample.py) | Sample + status types |
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
- The return value is not the trajectory. Samples are collected from the active `RolloutContext` (see [Samples](#samples)); the integrations register sources for you.

## Grader

```python
class Grader(ABC):
    def __init__(self, config: GraderConfig | None = None): ...
    @abstractmethod
    async def grade(self, ctx: GraderContext) -> Any: ...
```

[../osmosis_ai/rollout/grader.py](../osmosis_ai/rollout/grader.py)

- `ctx.get_samples()` returns the collected `dict[str, RolloutSample]` (**sync**).
- Attach rewards with `ctx.set_sample_reward(sample_id, reward)` — it raises `ValueError` for an unknown `sample_id` ([context.py](../osmosis_ai/rollout/context.py)).
- `ctx.label` carries the dataset row's label (the ground-truth string).

## Contexts

[../osmosis_ai/rollout/context.py](../osmosis_ai/rollout/context.py)

- `AgentWorkflowContext` — `prompt: list[dict]`, `config`.
- `HarborAgentWorkflowContext` — adds `environment` (Harbor `BaseEnvironment`) for `environment.exec()`, `environment.upload_file()`, etc. under `HarborBackend`.
- `GraderContext` — `label`, `samples`, plus `get_samples()` / `set_sample_reward()`. (It also has a `project_path` field, but no backend currently populates it — it is always `None`.)
- `RolloutContext` — ambient per-rollout context (chat completions URL, API key, rollout id). It is a context manager; the server enters it around execution. Local backends pass connection info directly; container runners read it from `OSMOSIS_CHAT_COMPLETIONS_URL` / `OSMOSIS_API_KEY` / `OSMOSIS_ROLLOUT_ID`. Fetch the current one with `get_rollout_context()`.

### Samples

The workflow does not return samples; instead a `SampleSource` is registered on the **ambient** `RolloutContext` (fetched with `get_rollout_context()`, not the `ctx` passed to `run`) and called lazily at collection time:

```python
from osmosis_ai.rollout import get_rollout_context

rollout_ctx = get_rollout_context()                  # the active RolloutContext
rollout_ctx.register_sample_source(name, source)     # name must be unique per rollout
samples = await rollout_ctx.get_samples()            # async -> {name: RolloutSample}
```

`OsmosisStrandsAgent` registers a source automatically (keyed by the agent `name`/`agent_id`), so most workflows never call `register_sample_source` directly.

`RolloutSample` ([types/sample.py](../osmosis_ai/rollout/types/sample.py)) fields: `id`, `messages`, `label`, `reward`, `remove_sample`, `metrics`, `extra_fields`.

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
- `ControllerAuth` ([server/auth.py](../osmosis_ai/rollout/server/auth.py)) supplies the bearer headers for callbacks.
- `ExecutionBackend` ([backend/base.py](../osmosis_ai/rollout/backend/base.py)) is the ABC; pick one:
  - `LocalBackend` ([backend/local/](../osmosis_ai/rollout/backend/local/)) — runs workflow + grader in-process. Re-exported from `osmosis_ai.rollout`. Used by the scaffold and eval.
  - `HarborBackend` ([backend/harbor/backend.py](../osmosis_ai/rollout/backend/harbor/backend.py)) — runs the agent inside a Harbor container; pairs with `HarborAgentWorkflowContext`. It is **not** re-exported (import `from osmosis_ai.rollout.backend.harbor.backend import HarborBackend`) and requires the external `harbor` dependency.

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
        for sample_id, sample in ctx.get_samples().items():
            reward = 1.0 if str(ctx.label) in str(sample.messages[-1]) else 0.0
            ctx.set_sample_reward(sample_id, reward)
```

For complete, runnable rollouts (local Strands, local OpenAI Agents, Harbor) see the [Osmosis-AI/workspace-template](https://github.com/Osmosis-AI/workspace-template) `rollouts/` directory.

## See also

- [architecture.md](./architecture.md) — protocol + execution model
- [troubleshooting.md](./troubleshooting.md) — timeouts, event-loop blocking, concurrency
