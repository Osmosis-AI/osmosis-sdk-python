# CLI reference

Installing the SDK provides a lightweight CLI as `osmosis` (aliases: `osmosis_ai`, `osmosis-ai`). The CLI loads `.env` from the current working directory via `python-dotenv`.

## Authentication

Credentials are stored at `~/.config/osmosis/credentials.json` (workspace-aware).

### osmosis auth login

Device code flow:

```bash
osmosis auth login
osmosis auth login --force
```

### osmosis auth logout

```bash
osmosis auth logout
osmosis auth logout --all
osmosis auth logout -y
```

### osmosis auth whoami

```bash
osmosis auth whoami
```

> Top-level `osmosis login`, `osmosis logout`, and `osmosis whoami` still work as hidden aliases during the transition.

### osmosis workspace

Manage workspaces interactively. Launches a TUI that shows your current workspace context and lets you switch workspace, browse training runs, datasets, and models.

```bash
osmosis workspace
```

In non-interactive environments, prints current context and exits.

## Rollout

### osmosis rollout serve

Start a v2 RolloutServer from a TOML file.

`osmosis rollout serve` requires the entrypoint module to expose a concrete `AgentWorkflow` and a concrete `Grader`. In practice you will usually also define a `GraderConfig`. If no grader is discoverable, serve fails before startup.

```bash
osmosis rollout serve serve.toml
osmosis rollout serve serve.toml --local
osmosis rollout serve serve.toml --skip-register
osmosis rollout serve serve.toml -p 9100 -H 127.0.0.1
osmosis rollout serve serve.toml --validate-only
```

**`serve.toml` shape (minimal):**

```toml
[serve]
rollout = "my_rollout"
entrypoint = "workflow.py"

[server]
port = 9000
host = "0.0.0.0"
log_level = "info"

[registration]
skip = false
# api_key = "optional-static-key"

[debug]
no_validate = false
# trace_dir = "./traces"
```

| Option | Description |
|--------|-------------|
| `-p` / `--port` | Override `[server].port` |
| `-H` / `--host` | Override `[server].host` |
| `--no-validate` | Skip backend validation |
| `--validate-only` | Validate the required workflow/grader pair and exit |
| `--log-level` | Override Uvicorn log level |
| `--skip-register` | Do not register with the platform |
| `--local` | No API key auth, no platform registration |

With `--local`, do not set `[registration].api_key` in the config file.

### osmosis rollout test

Smoke-test an `AgentWorkflow` against a dataset (no grader required). Uses the same execution path as `osmosis eval run`, with a generated config.

This subcommand is currently hidden from standard help output, so invoke it directly by name as `osmosis rollout test ...`.

```bash
osmosis rollout test -m _ -d data.jsonl --model gpt-5-mini
osmosis rollout test -m _ -d data.jsonl --model openai/gpt-5-mini --limit 5
```

See [Test mode](./test-mode.md) for required project layout and options.

### osmosis rollout list

Reserved; not implemented yet.

## Evaluation

### osmosis eval run

Evaluate using a TOML config. The workflow is loaded from the entrypoint module, and the grader is usually auto-discovered from that same module, so most configs do not need a separate `[grader]` table.

```bash
osmosis eval run eval.toml
osmosis eval run eval.toml --fresh
osmosis eval run eval.toml --retry-failed
osmosis eval run eval.toml --limit 20 --batch-size 4
osmosis eval run eval.toml -o ./results --log-samples
```

See [Eval mode](./eval-mode.md) for the full `[eval]`, `[llm]`, `[runs]`, `[baseline]`, and `[output]` sections.

### osmosis eval cache

```bash
osmosis eval cache dir
osmosis eval cache ls
osmosis eval cache ls --model gpt-4 --status completed
osmosis eval cache rm <task_id>
osmosis eval cache rm --all --yes
```

| Subcommand / option | Description |
|---------------------|-------------|
| `dir` | Print cache root |
| `ls` | List caches (`--model`, `--dataset`, `--status`) |
| `rm` | Delete by `task_id`, `--all`, or filters (`-y` skips prompt) |

### osmosis eval rubric

LLM-as-judge on a JSONL conversation file:

```bash
osmosis eval rubric -d data.jsonl \
  --rubric "Evaluate the assistant's helpfulness..." \
  --model openai/gpt-5.4
```

| Flag | Description |
|------|-------------|
| `-d` / `--data` | JSONL path (required) |
| `-r` / `--rubric` | Inline rubric or `@file.txt` (required) |
| `--model` | Judge model, LiteLLM form (required) |
| `-n` / `--number` | Runs per record |
| `-o` / `--output` | JSON results path |
| `--api-key` | Judge API key |
| `--timeout` | Seconds |
| `--score-min` / `--score-max` | Score range |

## See also

- [Test mode](./test-mode.md)
- [Eval mode](./eval-mode.md)
- [Dataset format](./datasets.md)
- [Troubleshooting](./troubleshooting.md)
