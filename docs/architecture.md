# Architecture

> Code-anchored map of the `osmosis_ai` package for developers. For platform concepts and CLI usage see [docs.osmosis.ai](https://docs.osmosis.ai).

## Package layout

```text
osmosis_ai/
├── cli/               # CLI framework + all command groups (Typer)
│   ├── main.py        # Entry point & command registration (osmosis_ai.cli.main:main)
│   ├── errors.py      # CLIError — the single error type used by every domain
│   ├── console.py     # Console (rich + plain fallback)
│   ├── output/        # Output context, result types, JSON/plain envelopes
│   └── commands/      # Thin Typer shells; delegate to platform/cli + eval
├── platform/          # Everything that talks to the Osmosis Platform API
│   ├── auth/          # Device-code login, credential store, HTTP client
│   ├── api/           # OsmosisClient
│   └── cli/           # Platform CLI business logic (no Typer registration)
├── rollout/           # Remote Rollout SDK (see rollout-sdk.md)
│   ├── agent_workflow.py  # AgentWorkflow ABC
│   ├── grader.py          # Grader ABC
│   ├── context.py         # RolloutContext / AgentWorkflowContext / GraderContext
│   ├── driver.py          # RolloutDriver / RolloutRunRequest — eval-facing execution contract
│   ├── http_driver.py     # optional concrete HTTP driver (`[eval-run]`)
│   ├── controller/        # callback store (core); listener + eval-proxy client (`[eval-run]`)
│   ├── server/            # optional generic FastAPI server (`[server]`) + ControllerAuth
│   ├── backend/           # ExecutionBackend ABC + Local / optional Harbor backend
│   ├── container/         # in-container agent + grader runner and its file contract
│   ├── trajectory/        # ATIF models, conversion, and best-effort persistence
│   ├── types/             # protocol.py, config.py, output.py, sample.py
│   ├── utils/             # framework-neutral helpers (errors, http, ttl_cache, …)
│   └── integrations/agents/  # Strands / OpenAI Agents adapters
├── eval/              # Eval helpers
│   └── rubric/        # evaluate_rubric() LLM-as-judge engine
├── templates/         # `osmosis template` recipe catalog + source resolution
├── packaging.py       # Build an installable wheel bundle from a rollout project
├── __init__.py        # Top-level exports (lazy __getattr__)
├── _imports.py        # Lazy-export + missing-extra helpers shared by every facade
├── _litellm_compat.py # LiteLLM import shim (used by eval/rubric/engine.py)
└── consts.py          # PACKAGE_VERSION
```

## Domain boundaries

- `cli/` — the CLI framework layer plus every command group. Files in [../osmosis_ai/cli/commands/](../osmosis_ai/cli/commands/) are thin shells that delegate to business logic; see [cli.md](./cli.md).
- `platform/` — anything that calls the Osmosis Platform API. Business-logic helpers (no Typer registration) live in [../osmosis_ai/platform/cli/](../osmosis_ai/platform/cli/).
- `rollout/` — the remote rollout protocol SDK: the `AgentWorkflow` + `Grader` abstraction and the framework-neutral execution core (`LocalBackend`, contexts, trajectory persistence, the in-container runner). The generic FastAPI server and the framework/back-end adapters are explicit optional modules gated behind extras; see [rollout-sdk.md](./rollout-sdk.md).
- `eval/` — `rubric/` powers `osmosis eval rubric`; see [eval.md](./eval.md). The workflow/grader loader that cloud `eval submit` / `train submit` preflight uses lives in [../osmosis_ai/platform/cli/rollout_entrypoint.py](../osmosis_ai/platform/cli/rollout_entrypoint.py).

## Key import paths

```python
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.console import Console
from osmosis_ai.platform.auth import load_credentials
from osmosis_ai.eval.rubric import evaluate_rubric, RubricResult
from osmosis_ai.rollout import AgentWorkflow, Grader, LocalBackend, SampleSource
from osmosis_ai.rollout.server import create_rollout_server
from osmosis_ai.rollout.backend.harbor import HarborBackend
from osmosis_ai.rollout.integrations.agents.strands import OsmosisStrandsAgent
from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent
from osmosis_ai.rollout.controller import (
    CallbackStore,
    CallbackListener,
    EvalProxyClient,
)
from osmosis_ai.rollout.http_driver import HttpRolloutDriver
```

`osmosis_ai.rollout` is **not** re-exported at the package top level — import it directly. Its public surface is framework-neutral core only; it does not export the server or Strands integration. `server`, `harbor`, `strands`, `openai-agents`, and `eval-run` each require their matching installation extra. The generic server has no Harbor dependency. `CallbackStore` is in-process and does not require `[eval-run]`; the localhost listener, eval-proxy client, and `HttpRolloutDriver` do.

## Lazy loading

CLI startup must stay fast (~150 ms vs ~1 s), so heavy dependencies load on first use, not at import time:

- **Top-level package** — [../osmosis_ai/__init__.py](../osmosis_ai/__init__.py) resolves rubric exports (`evaluate_rubric`, `RubricResult`, error types) through `__getattr__`. Only `__version__` is eager.
- **Command shells** — every file in [../osmosis_ai/cli/commands/](../osmosis_ai/cli/commands/) uses function-level imports for heavy deps (`rollout.*`, `platform.api.*`, `platform.cli.*`, `eval.*`, `cli.console`). Module-level imports stay light: `typer`, `cli.errors`, the lightweight `platform.constants`, and stdlib.
- **No eager `cli.main`** — [../osmosis_ai/cli/__init__.py](../osmosis_ai/cli/__init__.py) does not import `cli.main`, which prevents circular imports when rollout/server modules import `cli.console`. The entry point is `osmosis_ai.cli.main:main` directly.
- **`_litellm_compat.py`** stays at the package top level because `eval/rubric/` depends on it.

## Remote rollout protocol

The core design separates **LLM inference** (on the training cluster) from **agent logic** (on your RolloutServer). The controller (Traingate/slime) drives inference; your `AgentWorkflow` runs the agent and your `Grader` attaches rewards. Inference weights stay on the training cluster so PPO sees consistent model weights, while agent/tool code can run anywhere.

```mermaid
sequenceDiagram
    participant C as RolloutController
    participant S as RolloutServer (create_rollout_server)
    participant W as AgentWorkflow.run
    participant G as Grader.grade
    C->>S: POST /rollout (RolloutInitRequest)
    Note over S: schedules the execution task, then returns 202 immediately
    S->>W: backend.execute(ExecutionRequest)
    W->>C: POST chat_completions_url (messages + tools)
    C-->>W: LLM response (tool_calls)
    Note over W: repeat until done
    S->>C: POST completion_callback_url (RolloutCompleteRequest)
    S->>G: grade collected samples
    S->>C: POST grader_callback_url (GraderCompleteRequest)
```

Anchors:

- Server + endpoints + callbacks: [../osmosis_ai/rollout/server/app.py](../osmosis_ai/rollout/server/app.py) (`create_rollout_server`, `POST /rollout`, `GET /health`, `_handle_rollout`).
- Wire types: [../osmosis_ai/rollout/types/protocol.py](../osmosis_ai/rollout/types/protocol.py) (`RolloutInitRequest`, `RolloutCompleteRequest`, `GraderCompleteRequest`, `GraderStatus`). Callbacks use `controller_api_key`; the optional `llm_api_key` is the chat/proxy bearer. When `llm_api_key` is omitted (`None`) the server falls back to the controller key; an explicit empty string is rejected. `GET /rollout/{id}/status` is backend-authoritative (`LocalBackend` may report `UNKNOWN`).
- Execution contract: [../osmosis_ai/rollout/backend/base.py](../osmosis_ai/rollout/backend/base.py) — `ExecutionBackend.execute(request, on_workflow_complete, on_grader_complete)`, where the two callbacks are `ResultCallback` parameters (not methods).
- Sample/result types: [../osmosis_ai/rollout/types/sample.py](../osmosis_ai/rollout/types/sample.py) (`RolloutSample`, `RolloutStatus`, `RolloutErrorCategory`, `ExecutionRequest`, `ExecutionResult`).

The controller delivers results asynchronously via the two callback URLs, so it can manage many concurrent rollouts. `grader_callback_url` is optional; when omitted, grading is skipped.

### Eval path

The eval-facing contract is `RolloutDriver` / `RolloutRunRequest` / `RolloutOutcome` in [../osmosis_ai/rollout/driver.py](../osmosis_ai/rollout/driver.py). The optional HTTP implementation is [../osmosis_ai/rollout/http_driver.py](../osmosis_ai/rollout/http_driver.py), with a generic callback store and localhost listener under [../osmosis_ai/rollout/controller/](../osmosis_ai/rollout/controller/). The store exposes separate `wait_completion` and `wait_terminal` rendezvous so callers can await workflow completion separately from the terminal grader result. Eval supplies data + an LLM endpoint and consumes trace + reward, without caring whether execution was in-process or over HTTP.

## Backend validation

Cloud `eval submit` / `train submit` preflight validates paths and declared dependencies, then imports the rollout entrypoint once. There is no static validation layer beyond that import: the CLI does not scan the module namespace for `AgentWorkflow` or `Grader` classes, and the server does not inspect the backend it is given. Backend constructors establish their own invariants, so genuine misconfigurations surface as import-time errors during preflight; anything subtler surfaces as an ordinary Python error on the first rollout. Running an eval (`osmosis eval submit`) exercises the rollout end to end and is the intended smoke test before training.

## See also

- [rollout-sdk.md](./rollout-sdk.md) — the library API surface
- [cli.md](./cli.md) — CLI internals
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev workflow
