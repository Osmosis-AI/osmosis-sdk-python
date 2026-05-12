# Troubleshooting

Common errors when using the Osmosis SDK CLI and local evaluation.

## Installation

### Missing optional dependencies

| Error | Install |
|-------|---------|
| `No module named 'fastapi'` / `uvicorn` | `pip install osmosis-ai[server]` |
| `No module named 'pydantic_settings'` | `pip install osmosis-ai[server]` |
| `No module named 'pyarrow'` | `pip install pyarrow` or `pip install osmosis-ai[server]` |
| `No module named 'rich'` | `pip install osmosis-ai[server]` |
| `No module named 'litellm'` | Should ship with core; reinstall `osmosis-ai` |

```bash
pip install osmosis-ai[full]
pip install osmosis-ai[dev]   # + pytest, ruff, etc.
```

### Python version

Requires **Python 3.12+**. Check with `python --version`.

## Authentication

Credentials path: `~/.config/osmosis/credentials.json`.

### `osmosis auth login` fails

1. Ensure outbound HTTPS to the platform is allowed.
2. Retry with `osmosis auth login --force`.

### Token expiration

```
Not logged in. Run 'osmosis auth login' first.
```

Run `osmosis auth login` again before using platform commands that use the linked project workspace.

### Wrong linked workspace

From the project directory, link this project with the intended workspace:

```bash
osmosis project link --workspace <workspace-id-or-name> --yes
```

Confirm the workspace and Git Sync repository in the Osmosis Platform, then
re-run `osmosis project link --workspace <workspace-id-or-name> --yes` from the
matching local checkout.

## Eval server and grader issues

### Eval Fails Because 127.0.0.1:8000 Is Occupied

Controller-backed eval owns the rollout server process and uses fixed port `8000`. Stop the existing local server or wait for the other eval run to finish. Cached-only eval runs return cached results without touching port `8000`.

### Eval Server Starts But `/health` Never Passes

Check the user-server subprocess log. For `osmosis eval run`, it is stored next to the eval cache under `.osmosis/cache/eval/<sanitized-model>/<sanitized-dataset-stem>/user-server-<task_id>.log`; for example, `openai/gpt-5-mini` appears as `openai-gpt-5-mini`. Eval starts the server with `uv run python <entrypoint>` from `rollouts/<rollout>`, so imports that only worked when eval loaded code in-process must be changed to work from the rollout directory. The rollout folder must contain `pyproject.toml`.

### Missing Grader Rewards

The controller tracks sample IDs when `/chat/completions` reaches the sample-create point. The grader callback must include a sample entry for every controller-created sample ID. Normal samples need a non-null reward. Skipped or removed samples can set `remove_sample=true` with `reward=None`.

## Rubric (`osmosis eval rubric`)

### Missing API key

Set the provider-specific environment variable (e.g. `OPENAI_API_KEY`) or pass `--api-key`.

### Provider errors

Check quota, model name, and network. Increase `--timeout` if calls are slow.

## Dataset errors

### `DatasetParseError`

Unsupported extension or malformed file. Supported: `.parquet`, `.jsonl`, `.csv`.

### `DatasetValidationError`

Every row needs non-empty string `ground_truth`, `user_prompt`, and `system_prompt`. See [Dataset format](./datasets.md).

## Eval cache

### Lock contention

```
TimeoutError: Another eval with the same config is already running.
```

Wait for the other process, or:

```bash
export OSMOSIS_EVAL_LOCK_TIMEOUT=120
```

### Dataset changed mid-run

Use `--fresh` after mutating the dataset file.

### Stale cache format

Upgrade the package or delete the specific cache file after `osmosis eval cache ls`.

## Training run rollout failures

### All rollouts timeout / zero reward across the board

If a training run completes with `rollout/raw_reward = 0` and `rollout/response_len/mean = 0`, every rollout timed out before producing output. This usually means the LLM inference engine was overwhelmed with too many concurrent requests.

**Cause:** `rollout_batch_size` defaults to 64. With `n_samples_per_prompt = 8` that's 512 concurrent LLM calls hitting the rollout server simultaneously, which saturates the SGLang engine and causes every rollout to exceed `agent_workflow_timeout_s`.

**Fix:** Reduce `rollout_batch_size` in your training config:

```toml
[training]
n_samples_per_prompt = 8
rollout_batch_size = 8    # 8 × 8 = 64 concurrent calls instead of 512
```

If rollouts are still timing out with a smaller batch size (e.g. because your agent takes many turns), increase the timeout:

```toml
[training]
agent_workflow_timeout_s = 900   # 15 minutes instead of the default 7.5
```

### Some rollouts timeout (a few rows show zero reward / no output)

A small number of samples failing intermittently (2–5 rows out of 500+) indicates a resource contention issue on the rollout server rather than a total overload.

Common causes:
- **Event loop blocking**: synchronous calls inside an `async` rollout workflow (e.g. `mcp.list_tools_sync()`) freeze the uvicorn event loop. New HTTP requests from our trainer cannot get a 200 OK within the 30 s httpx connect timeout. Fix: wrap blocking calls in `asyncio.get_running_loop().run_in_executor(None, ...)`.
- **Subprocess exhaustion**: too many concurrent MCP subprocesses saturating OS limits. Set `ConcurrencyConfig(max_concurrent=64)` in your `AgentWorkflowConfig`.

## Training submission preflight

### Preflight failures

Run an eval smoke test before submitting:

```bash
osmosis eval run configs/eval/<run>.toml --limit 1
```

Then submit the training config:

```bash
osmosis train submit configs/training/<run>.toml
```

Training submission performs its own preflight checks before launching the
managed run. Fix reported rollout, config, dataset, or Git Sync issues before
submitting again.

## See also

- [CLI reference](./cli.md)
- [Dataset format](./datasets.md)
- [Eval](./eval.md)
