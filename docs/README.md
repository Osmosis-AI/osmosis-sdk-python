# Osmosis SDK documentation

## Project workflow

- Start from a Platform/Git Sync managed Osmosis project repo, or clone an existing Osmosis project.
- Run an eval smoke test: `osmosis eval run configs/eval/<config>.toml --limit 1`
- Submit training after Git Sync is connected: `osmosis train submit configs/training/<config>.toml`

## Workflow commands

- [Dataset format](./datasets.md) — Parquet / JSONL / CSV columns
- [Eval](./eval.md) — `osmosis eval run`, caching, pass@k
- [CLI reference](./cli.md) — all `osmosis` commands and options, including
  `train submit` with `[rollout.env]` / `[rollout.secrets]`
- [Troubleshooting](./troubleshooting.md)
