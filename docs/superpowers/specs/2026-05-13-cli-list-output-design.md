# CLI List Output Design

Date: 2026-05-13

## Context

Several `osmosis train` commands render too much implementation detail in tables
and do not prioritize the values users scan for first. Related list commands for
datasets and models have the same layout pressure: the human-facing name should
remain readable before IDs and metadata.

The current `train list` `Accuracy` column is also misleading. The platform web
training table displays `Reward`, sourced from the latest MLflow
`rollout/raw_reward` metric. The CLI list API currently returns `eval_accuracy`
and `reward_increase_delta`, but not the web table's `reward` value, so the SDK
cannot display the same value without an API change.

## Goals

- Make `train list` concise and scan-friendly.
- Show reward consistently with the platform web training table.
- Prefer complete names over complete IDs when terminal width is constrained.
- Convert human-facing timestamps to local time and mark list timestamp headers
  with the local timezone.
- Keep `--json` stable and machine-oriented.
- Keep `--plain` free of Rich markup and ANSI sequences.
- Reuse formatting logic across training runs, datasets, and models instead of
  duplicating per-command display code.
- Move complex Rich renderables and action hints outside detail tables.

## Non-Goals

- Do not change raw JSON timestamp values returned by serializers.
- Do not fetch per-run metrics from the SDK during `train list`.
- Do not redesign command names or pagination behavior.
- Do not change unrelated deployment, rollout, or eval list output unless shared
  renderer changes naturally affect them.

## Reward Source

The `Reward` value in `train list` will be the latest MLflow
`rollout/raw_reward` value, matching the web training table.

The monolith CLI API should add a nullable `reward` field to
`GET /api/cli/training-runs`. It should compute that value with the same batch
MLflow lookup used by the web table. The SDK will add `reward` to its
`TrainingRun` model and public serializer. If the API returns no `reward`, the
CLI will show an empty reward cell while preserving JSON compatibility.

## Shared List Formatting

Extend the CLI output result model with reusable display metadata:

- A primary text column that should be preserved first under narrow terminal
  widths.
- Human labels that can differ from serializer keys. For these commands the
  primary name column label should be `Name`:
  - training run: `name`
  - dataset: `file_name`
  - model: `model_name`
- Optional display-only values for Rich and plain output.
- Optional post-list hints shown only in Rich human output.

Rich table behavior should use these metadata fields to let the primary name
column wrap or retain width before secondary columns. IDs should not force the
table to preserve full width. Plain output should use the same display values
without Rich markup. JSON output should ignore display metadata and emit the raw
serializer envelope.

## Time Formatting

Add a shared local-time formatter for human output:

- `format_local_date`: compact list date/time for `Created` columns.
- `format_local_datetime`: detailed detail-view timestamp for `Created`,
  `Started`, and `Completed`.
- `local_timezone_label`: short label such as `PDT` when available.

List column headers should include the timezone, for example `Created (PDT)`.
If parsing fails or the timestamp lacks enough information, retain the current
fallback behavior rather than failing the command.

## Status Formatting

Keep separate status classification for training runs and general platform
entities, but expose them through a shared display helper:

- Training run success/in-progress/error/stopped statuses use the existing run
  status mapping.
- Dataset and model statuses use the existing entity status mapping.
- Rich output applies color.
- Plain output strips Rich markup.
- JSON output emits raw status strings.

## Command Designs

### `osmosis train list`

Rich/plain display columns:

- `Name`
- `Status`
- `Reward`
- `Created (<local TZ>)`

The command should not display ID or model columns in the human table. JSON
should still include the complete serialized run, including ID, model fields,
`eval_accuracy`, `reward_increase_delta`, and `reward`.

Rich output should show a dim post-list hint:

`Use osmosis train status <name> or osmosis train metrics <name> for details.`

### `osmosis train status`

The primary detail table should contain scalar run metadata. `Created`,
`Started`, and `Completed` should use detailed local time when parseable.

Checkpoints should render outside the detail table as a separate Rich table or
plain lines. The deploy guidance should render as a dim hint outside the table.
JSON should continue to include `checkpoints` in the detail data.

### `osmosis train metrics`

The summary table should contain scalar metadata and metric summary values only.

Metric trends should render outside the summary table in Rich mode. They should
not be stored as a `DetailField` value because Rich renderables stringify poorly
inside the generic detail table.

Saved output path and save warnings should render as post-detail hints. JSON
should keep `output_path` and `save_warning` fields.

### `osmosis dataset list`

Rich/plain display columns:

- `Name`
- `Status`
- `Size`
- `Created (<local TZ>)`

The full dataset ID remains in JSON. The human table should not let the ID
determine the width budget. Dataset status and size formatting should use the
shared display helpers.

### `osmosis model list`

Rich/plain display columns:

- `Name`
- `Status`
- `Base`
- `Created (<local TZ>)`

The full model ID remains in JSON. The human table should prioritize model name
over ID and metadata. Status formatting should use the shared entity status
helper.

## Data Flow

1. Monolith list API returns training runs with nullable `reward`.
2. SDK API model parses `reward` into `TrainingRun`.
3. SDK serializer emits raw training run fields for JSON.
4. Command handlers build `display_items` with local-time and styled status
   values.
5. Renderer emits JSON from raw items, plain from normalized display values, and
   Rich from display values plus post-list/post-detail renderables.

## Testing

- Add monolith tests for `GET /api/cli/training-runs` returning `reward` from
  MLflow batch rewards.
- Add SDK API model tests for parsing `reward`.
- Add SDK serializer tests for emitting `reward`.
- Add SDK command tests for `train list` columns and display values.
- Add renderer tests for primary-column preservation metadata and post-render
  hints/sections.
- Add tests that Rich status display does not leak markup into `--plain`.
- Add tests for local timestamp formatting with parseable and invalid inputs.
- Add regression tests that Rich renderables in metrics trends are not embedded
  as stringified `DetailField` values.

## Risks

- The monolith and SDK changes must land in compatible order. The SDK should
  tolerate missing `reward` to avoid breaking against older API deployments.
- Terminal width behavior depends on Rich's layout algorithm. Tests should focus
  on command metadata and representative narrow rendering rather than exact box
  drawing.
- Timezone abbreviations can vary by host environment. Tests should control the
  timezone or assert against injected/known timezone behavior.
