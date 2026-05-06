# CLI reference

Installing the SDK provides a lightweight CLI as `osmosis` (aliases: `osmosis_ai`, `osmosis-ai`). The CLI loads `.env` from the current working directory via `python-dotenv`.

## Authentication

Credentials are stored at `~/.config/osmosis/credentials.json`.

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

`whoami` verifies the active credentials and reports the authenticated account.
It does not inspect local project links. Use `osmosis project info` for the
current project link or `osmosis project list` for the local link table.

> Top-level `osmosis login`, `osmosis logout`, and `osmosis whoami` still work as hidden aliases during the transition.

### osmosis workspace

Open the interactive workspace browser. The TUI lets you choose an accessible workspace, then browse training runs, datasets, and models.

```bash
osmosis workspace
```

In non-interactive environments, use a specific subcommand such as `osmosis workspace list`, `osmosis workspace create <name>`, or `osmosis workspace delete <name> --yes`. To link this project with a workspace for platform commands, use `osmosis project link --workspace <workspace-id-or-name>`.

## Project

### osmosis project link

Link this project with an Osmosis workspace. The workspace must have a Git
Sync connected repository, and the command must be run from a checkout whose
`origin` remote matches that repository. Platform commands resolve their
workspace from the current project.

```bash
osmosis project link --workspace <workspace-id-or-name>
osmosis project link --workspace <workspace-id-or-name> --yes
```

The project mapping is stored in `~/.osmosis/config.json`.

### osmosis project info

Show the local workspace link for the current project. This reads local metadata
by default and only contacts the platform when `--refresh` is passed.

```bash
osmosis project info
osmosis project info --refresh
```

### osmosis project list

List project-to-workspace links stored on this machine. This command reads the
local mapping table and does not require authentication.

```bash
osmosis project list
osmosis project list --all-platforms
```

For CI:

```bash
export OSMOSIS_TOKEN=<token>
osmosis project link --workspace <workspace-id-or-name> --yes
osmosis train submit configs/training/default.toml --yes
```

To adopt an existing repository as an Osmosis project:

```bash
git clone <repo_url>
cd <repo>
osmosis init --here <name>
osmosis project link --workspace <workspace-id-or-name>
```

### osmosis project validate

Validate the canonical layout of a local Osmosis project (the directory created
by `osmosis init`).

```bash
osmosis project validate
osmosis project validate ./path/to/project
```

The command checks for `.osmosis/project.toml` and the required `rollouts/`,
`configs/training/`, `configs/eval/`, and `data/` directories.

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

The command only accepts configs under these canonical project paths:

- `configs/eval/<name>.toml`
- `configs/training/<name>.toml`

### osmosis rollout list

List rollouts in the linked project workspace.

```bash
osmosis rollout list
osmosis rollout list --limit 50
osmosis rollout list --all
```

## Evaluation

### osmosis eval run

Evaluate using a TOML config. The workflow is loaded from the entrypoint module, and the grader is usually auto-discovered from that same module, so most configs do not need a separate `[grader]` table. If the grader lives elsewhere, set `[grader].module` and optional `[grader].config`.

`osmosis eval run` expects the config file to live under `configs/eval/` inside a
structured Osmosis project.

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
