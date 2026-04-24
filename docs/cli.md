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

### osmosis rollout validate

Validate the rollout entrypoint referenced by a training or eval config.

`osmosis rollout validate` requires the resolved entrypoint module to expose a
concrete `AgentWorkflow` and a concrete `Grader` (unless an eval config
provides an explicit grader override).

```bash
osmosis rollout validate configs/eval/my-rollout.toml
osmosis rollout validate configs/training/my-run.toml
```

The command only accepts configs under these canonical workspace paths:

- `configs/eval/<name>.toml`
- `configs/training/<name>.toml`

### osmosis rollout list

List rollouts in the current workspace.

```bash
osmosis rollout list
osmosis rollout list --limit 50
osmosis rollout list --all
```

## Evaluation

### osmosis eval run

Evaluate using a TOML config. The workflow is loaded from the entrypoint module, and the grader is usually auto-discovered from that same module, so most configs do not need a separate `[grader]` table. If the grader lives elsewhere, set `[grader].module` and optional `[grader].config`.

`osmosis eval run` expects the config file to live under `configs/eval/` inside a
structured Osmosis workspace.

```bash
osmosis eval run configs/eval/my-rollout.toml
osmosis eval run configs/eval/my-rollout.toml --fresh
osmosis eval run configs/eval/my-rollout.toml --retry-failed
osmosis eval run configs/eval/my-rollout.toml --limit 20 --batch-size 4
osmosis eval run configs/eval/my-rollout.toml -o ./results --log-samples
```

See [Eval](./eval.md) for the full `[eval]`, `[llm]`, `[runs]`, `[baseline]`, and `[output]` sections.

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

- [Eval](./eval.md)
- [Dataset format](./datasets.md)
- [Troubleshooting](./troubleshooting.md)
