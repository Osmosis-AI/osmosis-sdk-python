# Osmosis Project

This is a **structured Osmosis project**. Do not invent a different
top-level layout.

## Project contract

- Required paths:
  - `.osmosis/project.toml`
  - `rollouts/`
  - `configs/training/`
  - `configs/eval/`
  - `data/`
- New rollouts live in `rollouts/<name>/`.
- The canonical rollout entrypoint is `rollouts/<name>/main.py`.
- Eval configs live in `configs/eval/<name>.toml`.
- Training configs live in `configs/training/<name>.toml`.
- Local training guidance lives in `.osmosis/research/program.md`.
- Local cache state lives in `.osmosis/cache/` and should not be treated as source.
- Do not create new top-level directories unless the user explicitly asks.

## Rollout contract

- Each rollout entrypoint must expose exactly one concrete `AgentWorkflow`.
- Local eval and managed training require a concrete `Grader` in the rollout
  server. Eval configs do not support `[grader]` overrides.
- Tools should be async Python functions with type hints and docstrings.
- `Grader.grade` must be async and return a float in `[0.0, 1.0]`.
- Before `osmosis train submit`, validate the project and run a local eval.

## Environment variables and secrets

Training configs can inject environment variables into the rollout container via
two optional TOML sections:

```toml
[rollout.env]
# Literal values baked into the config — visible in this file.
# Do NOT store secrets here.
LOG_LEVEL = "INFO"

[rollout.secrets]
# Maps env-var name → workspace environment_secret *record name*.
# The platform resolves the actual value server-side; it never appears
# in this file or in transit.
# Pre-register secrets at /:orgName/secrets before submitting.
OPENAI_API_KEY = "openai-api-key"
```

- Both sections are optional; omit entirely if not needed.
- Keys must match `^[A-Z_][A-Z0-9_]*$`.
- The same key cannot appear in both sections.
- Reserved names (`GITHUB_CLONE_URL`, `GITHUB_TOKEN`, `ENTRYPOINT_SCRIPT`,
  `REPOSITORY_PATH`, `TRAINING_RUN_ID`, `ROLLOUT_NAME`, `ROLLOUT_PORT`) are
  forbidden in both sections.
- Inside the container, all injected vars are available via `os.environ`.

## AI skills

Detailed workflow guidance lives in the **`osmosis` agent plugin**:

| Skill | What it does |
| --- | --- |
| `plan-training` | Turn a vague task into a concrete local training plan. |
| `create-rollouts` | Create or adapt rollouts, graders, entrypoints, and baseline eval configs. |
| `evaluate-rollouts` | Run local evals, compare baselines, and iterate with data. |
| `debug-rollouts` | Diagnose rollout, grader, config, dataset, or preflight failures. |
| `submit-training` | Prepare a training config and submit it safely. |

### Enabling the plugin

- **Claude Code** — `.claude/settings.json` in this project registers the
  plugin automatically; on first open, Claude Code prompts to install.
- **Cursor** — Settings → Rules → "Add Remote Rule" → paste the plugin repo
  URL (skills render as Remote Rules in Cursor).
- **Codex** — Run `codex plugin marketplace add {plugin_repo}` once, then
  `codex plugin install {plugin_marketplace}`.

The plugin repo is configured in `.claude/settings.json`. Check that file
before modifying plugin state.

## CLI output

- The commands below use the default rich output for interactive human sessions.
- For AI agents or automation, prefer `osmosis --json ...` for structured output
  or `osmosis --plain ...` for low-noise text.

## Common commands

```bash
osmosis project validate
osmosis rollout validate configs/eval/<name>.toml
osmosis rollout validate configs/training/<run>.toml
osmosis eval run configs/eval/<name>.toml
osmosis train submit configs/training/<run>.toml
osmosis train status <run-name>
```
