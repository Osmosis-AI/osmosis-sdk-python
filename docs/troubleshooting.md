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
Not logged in. Run 'osmosis auth login' first, or use --skip-register for local testing.
```

Run `osmosis auth login` or use `osmosis rollout serve ... --local` / `--skip-register` when you do not need registration.

### Wrong workspace

Run `osmosis workspace` to inspect or switch context.

## Reward and grader issues

Decorators like `@osmosis_reward` still apply to classic reward functions used in other platform flows. For **`osmosis eval run`**, scoring comes from your **Grader** implementation discovered next to the workflow — see [Eval mode](./eval-mode.md).

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

## Rollout server

### Validation failures

Run a one-shot check:

```bash
osmosis rollout serve serve.toml --validate-only
```

Fix reported workflow/grader issues before binding a port.

`osmosis rollout serve` requires a concrete `Grader` discoverable from the entrypoint module. If validation reports that no grader was found, add one there; in most projects you will also define a `GraderConfig`.

### Not logged in for registration

Either `osmosis auth login` or start with `--skip-register` / `--local`.

### Missing FastAPI

```
ImportError: ... Install ... osmosis-ai[server]
```

### Public host / IP detection

For registration, the process may need a reachable host. Set it with `-H` / `--host` on `osmosis rollout serve`, or in `[server].host` inside `serve.toml`. For local debugging without platform registration, use `--local`.

## See also

- [CLI reference](./cli.md)
- [Dataset format](./datasets.md)
- [Eval mode](./eval-mode.md)
