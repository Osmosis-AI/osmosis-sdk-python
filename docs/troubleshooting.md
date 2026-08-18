# Troubleshooting (engineering)

> Engineering-level failure modes when building rollouts and running evals. Install, login, and workspace-setup basics live at [docs.osmosis.ai](https://docs.osmosis.ai). One entry fact: the SDK requires **Python 3.12+** and the server extra (`pip install "osmosis-ai[server]"`) to run a rollout server (scaffold one with `osmosis rollout init <name>`, then run `python rollouts/<name>/main.py`).

## Rollout timeouts

The controller sends per-rollout `agent_timeout_sec` / `grader_timeout_sec` in the `RolloutInitRequest` ([../osmosis_ai/rollout/types/protocol.py](../osmosis_ai/rollout/types/protocol.py)), and both backends enforce them. The Harbor backend applies them per execution via `override_timeout_sec` ([../osmosis_ai/rollout/backend/harbor/backend.py](../osmosis_ai/rollout/backend/harbor/backend.py)). `LocalBackend` wraps the two phases in independent `asyncio.timeout()` scopes — the agent deadline around `AgentWorkflow.run()`, the grader deadline around `Grader.grade()` — so a workflow that finishes just inside its budget still gets its full grading window ([../osmosis_ai/rollout/backend/local/backend.py](../osmosis_ai/rollout/backend/local/backend.py)).

An expired deadline is a normal terminal result, not a crash: the rollout is reported with `err_category=timeout` and `err_message="workflow exceeded its <N>s deadline"`, and the concurrency slot is released. A `TimeoutError` your own code raises keeps its own message and is still categorized as `RolloutErrorCategory.TIMEOUT` ([../osmosis_ai/rollout/utils/errors.py](../osmosis_ai/rollout/utils/errors.py)). `None` means the controller sent no deadline, and that phase runs unbounded.

The deadline cancels a *cooperatively cancellable* workflow. Code that swallows `CancelledError`, or that blocks the event loop in synchronous work, still runs to completion — the rollout is reported as a timeout either way, but the slot is not freed until that code returns.

### All rollouts time out / zero reward across the board

If a training run completes with `rollout/raw_reward = 0` and `rollout/response_len/mean = 0`, every rollout timed out before producing output — usually the inference engine was overwhelmed by too many concurrent requests.

**Cause:** a high `rollout_batch_size`. With `rollout_batch_size = 64` and `n_samples_per_prompt = 8`, the controller fires `64 × 8 = 512` concurrent calls at the rollout server, saturating the SGLang engine so every rollout exceeds its timeout.

**Fix:** lower `rollout_batch_size` (a `[training]` field owned by the backend):

```toml
[training]
n_samples_per_prompt = 8
rollout_batch_size = 8    # 8 x 8 = 64 concurrent calls instead of 512
```

If rollouts still time out with a smaller batch (e.g. a long multi-turn agent), raise the timeout:

```toml
[training]
agent_workflow_timeout_s = 900   # 15 minutes instead of the default 7.5
```

### A few rollouts time out intermittently

2–5 failing rows out of 500+ points at resource contention on the rollout server, not a total overload. Two common causes:

- **Event-loop blocking** — a synchronous call inside an `async` workflow (e.g. `mcp.list_tools_sync()`) freezes the uvicorn event loop, so new HTTP requests can't get a `200 OK` within the trainer's connect timeout. Wrap blocking work off the loop:

  ```python
  import asyncio

  tools = await asyncio.get_running_loop().run_in_executor(None, mcp.list_tools_sync)
  ```

- **Subprocess exhaustion** — too many concurrent MCP subprocesses saturating OS limits. Cap in-flight executions via `ConcurrencyConfig` ([../osmosis_ai/rollout/types/config.py](../osmosis_ai/rollout/types/config.py)):

  ```python
  MyWorkflowConfig(name="my-rollout", concurrency=ConcurrencyConfig(max_concurrent=64))
  ```

## Backend validation

Cloud `osmosis eval submit` / `osmosis train submit` validate rollout paths and dependencies, then import the entrypoint once. The CLI does not infer backend requirements by scanning for workflow or grader classes, and there is no separate validation step: errors raised while the module constructs its backend (bad import strings, rejected configs) surface through submit preflight, and anything beyond that surfaces on the first rollout. Run an eval first — it exercises the workflow, grader, and server end to end and is the intended smoke test before training.

## Dataset validation

Local validation requires at least 4 rows and a `.csv` / `.jsonl` / `.parquet` extension. Without a `metadata` column, the dataset must contain `user_prompt` and `ground_truth` (`label` is accepted as an alias); with a `metadata` column, every row must contain a valid, non-empty JSON object. `system_prompt` is always optional, and JSONL rows must use identical top-level fields. See [datasets.md](./datasets.md) for the full contract.

## Rubric (`osmosis eval rubric` / `evaluate_rubric`)

- `MissingAPIKeyError` — set the provider env var (e.g. `OPENAI_API_KEY`) or pass `api_key` programmatically. The CLI still accepts `--api-key` for compatibility, but it is deprecated and emits a `DEPRECATION` warning; prefer the provider environment variable.
- `ModelNotFoundError` — wrong model identifier or no account access.
- `ProviderRequestError` — quota, rate limit, network, or a non-JSON model response; raise the `timeout` for slow providers.

See [eval.md](./eval.md) for the API and error hierarchy.

## See also

- [architecture.md](./architecture.md) — execution model and the protocol
- [rollout-sdk.md](./rollout-sdk.md) — workflow/grader/config API
