# Evaluation run output download contract

The eval download implementation lives in [`platform/cli/run_download.py`](../osmosis_ai/platform/cli/run_download.py). Its Typer shell remains thin and lives in [`cli/commands/eval.py`](../osmosis_ai/cli/commands/eval.py).

## Commands

```text
osmosis eval download NAME_OR_ID
  --type metrics,trajectories|artifacts|logs|all
  --rows 3,7,10-20
  -o, --output ROOT
  --overwrite
  -y, --yes
```

`--type` replaces the default selection and defaults to `metrics,trajectories`. A row selection includes every run for each selected row.

## Platform routes

After resolving a run name or ID, the SDK uses two eval routes:

```text
GET  /api/cli/eval-runs/[id]/samples/manifest?types=&rows=
POST /api/cli/eval-runs/[id]/samples/download-urls
```

The manifest returns `{files: [{token?, path, size}], totals}`. `path` is the final path relative to the local run root, and `token` is an opaque server handle (a rollout id or an export snapshot token). URL requests contain at most 500 `{token, path}` items. The platform derives full S3 keys server-side and returns 15-minute presigned GET URLs; the SDK never accepts raw object keys.

## Fixed local layout

```text
.osmosis/
├── evals/<run-name>/
│   ├── metrics.json
│   ├── summary.jsonl
│   ├── trajectories/row_3_run_0.json
│   ├── artifacts/row_3_run_0/logs/agent.log
│   └── logs.txt
└── metrics/                       # legacy eval exports; never deleted
```

`--output` relocates the run root; filenames and subdirectories below it do not change. Rich-mode `eval info` uses the same resolver and writes the same run-scoped `metrics.json` path. Names that require filesystem sanitization gain a stable run-ID suffix so distinct runs never share a local directory. Training download and training metrics-path migration are intentionally out of scope until the platform routes are ready; `train info` keeps its existing export behavior for now.

## Transfer behavior

- Files with a matching local size are skipped unless `--overwrite` is set.
- Remaining transfers over 100 MiB require confirmation; `--yes` skips it.
- Presigned URLs are requested batch-by-batch, with at most 500 files per call.
- Eight files download concurrently into sibling `*.partial` files, then move atomically into place after the manifest size is verified.
- Each file is retried up to three times with exponential backoff. Failures do not stop other files; the command reports every failed path and exits partial so re-running retries only missing or mismatched files.
- Manifest paths outside the fixed layout and reserved `manifest.json` entries are rejected defensively. The platform remains responsible for authoritative path validation and scoped S3 listing.

## Metrics export migration

New rich-mode eval info exports use `.osmosis/evals/<name>/metrics.json`. Existing `.osmosis/metrics/` files are left untouched. The CLI emits a one-release migration notice after saving.
