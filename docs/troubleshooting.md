# Troubleshooting

Common errors when using the Osmosis SDK CLI.

## Installation

### Missing optional dependencies

| Error | Install |
|-------|---------|
| `No module named 'fastapi'` / `uvicorn` | `pip install osmosis-ai[server]` |
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

Run `osmosis auth login` again before using platform-scoped commands from a
workspace directory.

## Workspace Directory Flow

Create or open a workspace in the Osmosis Platform, clone the repository created there,
then run CLI commands from that workspace directory.

```bash
git clone <repo-url>
cd <repo>
osmosis auth login
osmosis doctor
osmosis template apply multiply              # or add your rollout under rollouts/
cp configs/training/default.toml configs/training/<run>.toml
$EDITOR configs/training/<run>.toml          # set rollout, dataset, and model_path
git add rollouts configs data
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the workspace directory's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for commands scoped by the workspace directory.

### Wrong workspace directory

Confirm that the workspace directory's `origin` remote matches the repository created for
the intended Osmosis Platform workspace. If it does not, clone the correct
repository and rerun the command from that workspace directory.

## Rubric (`osmosis eval rubric`)

### Missing API key

Set the provider-specific environment variable (e.g. `OPENAI_API_KEY`) or pass `--api-key`.

### Provider errors

Check quota, model name, and network. Increase `--timeout` if calls are slow.

## Dataset errors

### Dataset validation fails

Unsupported extension or malformed file. Supported: `.parquet`, `.jsonl`, `.csv`.

Every row needs non-empty string `ground_truth`, `user_prompt`, and `system_prompt`. See [Dataset format](./datasets.md).

## Training run rollout failures

### All rollouts timeout / zero reward across the board

If a training run completes with `rollout/raw_reward = 0` and `rollout/response_len/mean = 0`, every rollout timed out before producing output. This usually means the LLM inference engine was overwhelmed with too many concurrent requests.

**Cause:** A high configured `rollout_batch_size` can create too many concurrent LLM calls. For example, `rollout_batch_size = 64` with `n_samples_per_prompt = 8` sends 512 concurrent calls to the rollout server, which can saturate the SGLang engine and cause every rollout to exceed `agent_workflow_timeout_s`.

**Fix:** Reduce `rollout_batch_size` in your training config:

```toml
[training]
n_samples_per_prompt = 8
rollout_batch_size = 8    # 8 x 8 = 64 concurrent calls instead of 512
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

Submit an evaluation run before a training run:

```bash
osmosis eval submit configs/eval/<run>.toml
```

Then submit the training run config:

```bash
osmosis train submit configs/training/<run>.toml
```

Training run submission performs its own preflight checks before launching the
managed run. Fix reported rollout, config, dataset, or repository scope issues before
submitting again.

## See also

- [CLI reference](./cli.md)
- [Dataset format](./datasets.md)
- [Eval](./eval.md)
