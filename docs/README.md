# Osmosis SDK documentation

## Project Flow

Create the project in the Osmosis Platform, clone the repository created there,
then run CLI commands from that checkout.

```bash
git clone <repo-url>
cd <repo>
osmosis auth login
osmosis project doctor
osmosis template apply multiply              # or add your rollout under rollouts/
cp configs/training/default.toml configs/training/<run>.toml
$EDITOR configs/training/<run>.toml          # set rollout, dataset, and model_path
git add rollouts configs data research
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the checkout's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for repo-scoped commands.

## Workflow commands

- [Dataset format](./datasets.md) — Parquet / JSONL / CSV columns
- [Eval](./eval.md) — `osmosis eval run`, caching, pass@k
- [CLI reference](./cli.md) — all `osmosis` commands and options, including
  `train submit` with `[rollout.env]` / `[rollout.secrets]`
- [Troubleshooting](./troubleshooting.md)
