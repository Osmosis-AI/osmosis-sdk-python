# Osmosis SDK documentation

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
git add rollouts configs data research
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the workspace directory's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for commands scoped by the workspace directory.

## Workflow commands

- [Dataset format](./datasets.md) — Parquet / JSONL / CSV columns
- [Eval](./eval.md) — `osmosis eval run`, caching, pass@k
- [CLI reference](./cli.md) — all `osmosis` commands and options, including
  `train submit` with `[rollout.env]` / `[rollout.secrets]`
- [Troubleshooting](./troubleshooting.md)
