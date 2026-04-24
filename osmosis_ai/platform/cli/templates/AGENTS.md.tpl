# Osmosis Workspace

This is a **structured Osmosis workspace**. Do not invent a different
top-level layout.

## Workspace contract

- Required paths:
  - `.osmosis/workspace.toml`
  - `.osmosis/research/`
  - `rollouts/`
  - `configs/training/`
  - `configs/eval/`
  - `data/`
- New rollouts live in `rollouts/<name>/`.
- The canonical rollout entrypoint is `rollouts/<name>/main.py`.
- Eval configs live in `configs/eval/<name>.toml`.
- Training configs live in `configs/training/<name>.toml`.
- Local experiment guidance lives in `.osmosis/research/program.md`.
- Experiment logs and planning artifacts live in `.osmosis/research/`.
- Do not create new top-level directories unless the user explicitly asks.

## Rollout contract

- Each rollout entrypoint must expose exactly one concrete `AgentWorkflow`.
- Local eval and managed training require a concrete `Grader` as well (or an
  explicit grader override where supported).
- Tools should be async Python functions with type hints and docstrings.
- `Grader.grade` must be async and return a float in `[0.0, 1.0]`.
- Before `osmosis train submit`, validate the workspace and run a local eval.

## AI skills

Detailed workflow guidance lives in the **`osmosis` agent plugin**:

| Skill | What it does |
| --- | --- |
| `vibe-train` | Orchestrate the full task-to-training workflow. |
| `bootstrap-rollout` | Create a baseline rollout, grader, and config set in canonical paths. |
| `iterate-rollout` | Run short local experiment loops driven by eval results. |
| `create-rollout-server` | Make a rollout entrypoint valid for Osmosis-managed hosting. |
| `curate-dataset` | Build or refine local datasets under `data/`. |
| `debug-rollout` | Diagnose rollout, grader, config, or eval failures. |
| `launch-training` | Prepare a training config and submit it safely. |

### Enabling the plugin

- **Claude Code** — `.claude/settings.json` in this workspace registers the
  plugin automatically; on first open, Claude Code prompts to install.
- **Cursor** — Settings → Rules → "Add Remote Rule" → paste the plugin repo
  URL (skills render as Remote Rules in Cursor).
- **Codex** — Run `codex plugin marketplace add {plugin_repo}` once, then
  `codex plugin install {plugin_marketplace}`.

The plugin repo is configured in `.claude/settings.json`. Check that file
before modifying plugin state.

## Common commands

```bash
osmosis workspace validate
osmosis rollout validate configs/eval/<name>.toml
osmosis rollout validate configs/training/<run>.toml
osmosis eval run configs/eval/<name>.toml
osmosis train submit configs/training/<run>.toml
osmosis train status <run-name>
```
