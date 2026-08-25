# Changelog

This file records changes to `osmosis-ai`. For earlier versions, see [GitHub Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases).

## 0.3.1 - 2026-08-24

### Breaking Changes

- `RolloutDriver.run` now takes a single `RolloutRunRequest` argument; update custom drivers and callers to pass the request object ([#307](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/307)).
- Removed `ExecutionBackend.max_concurrency` and the import-time `osmosis_ai.platform.auth.PLATFORM_URL`; use the rollout server's `/health` capacity and `get_platform_url()` for the active platform URL ([#315](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/315)).
- Removed unused server-owned fields from the public API record types; stop reading `UploadInfo.s3_key` / `.upload_id`, `DatasetFile.df_stats` / `.organization_id`, `TrainingRunMetrics.training_run_id`, `EvalRunMetrics.eval_run_id`, `RolloutInfo.last_synced_at`, and `TrainingRunCheckpoints.training_run_id` ([#315](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/315)).

### Added

- Added crash-safe `osmosis eval run` through the new `eval` extra for local evaluation from the same TOML used by managed runs, with dataset slicing, resumable output, uv-managed rollout environments, Local and Harbor Docker backends, readable run names, official OpenAI Responses routing, bounded admission, and orphan-server cleanup ([#307](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/307), [#310](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/310), [#316](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/316), [#317](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/317), [#318](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/318), [#321](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/321), [#322](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/322), [#323](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/323)).
- Added server-authoritative, idempotent `osmosis eval upload <run-dir>` and `eval run --upload` to import completed local runs without launching a managed evaluation ([#319](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/319), [#321](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/321)).

### Changed

- Standardized the CLI machine contract so `--json` and `--plain` never prompt, JSON errors use stable `{code, message, details}` envelopes on stderr, machine-readable warnings use JSON Lines, and non-finite values cannot produce invalid JSON ([#304](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/304)).
- Platform-scoped logins now persist in the operating-system keyring across directories and environments, keep credentials after HTTP 401 responses, and validate non-production environment tokens against `OSMOSIS_TOKEN_PLATFORM_URL` before network access ([#320](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/320)).

### Fixed

- Aligned Mini SWE-agent benchmark credential validation with the Platform while preserving Cursor CLI harness-key requirements ([#306](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/306)).
- `osmosis dev server up` now prints the one-time API key returned by the Platform so the provisioned server can be used immediately ([#305](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/305)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.0...v0.3.1)

## 0.3.0 - 2026-08-11

### Breaking Changes

- Each rollout now produces exactly one `RolloutSample` and reward through rollout-scoped URLs; migrate `samples`, `register_sample_source()`, and `set_sample_reward()` to `sample`, `set_sample_source()`, and `set_reward()`, remove per-call routing headers, and adopt the single-sample artifact contract ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Runtime integrations now use the `server`, `strands`, `openai-agents`, `harbor`, `rubric`, and `parquet` extras with explicit feature imports; update `AgentWorkflow.run()` to return one message history, replace `save_trajectories()` with `save_trajectory()`, and remove dependencies on the deleted `osmosis_ai.eval.common` and `osmosis_ai.rollout.validator` modules ([#270](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/270), [#277](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/277)).
- `HarborBackend` now refers to the container-native implementation previously named `HarborBackendV2`; the legacy backend, its constructor arguments, `OsmosisInstalledAgent`, and `HarborAgentWorkflowContext` are removed ([#272](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/272), [#291](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/291), [#292](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/292)).
- Datasets now use one uniform prompt or metadata schema across every row, and benchmark detail consumers must replace `required_secret_names` with `requires_judge_api_key` ([#282](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/282), [#290](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/290)).

### Added

- Added benchmark catalog, submission, run inspection, logs, cancellation, and output downloads through the `osmosis benchmark` CLI ([#265](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/265)).
- Added the container-native Harbor backend with installable workflow bundles, native agents, template and dataset task modes, prewarming, lifecycle diagnostics, artifact collection, admission control, status polling, and cancellation ([#272](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/272), [#285](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/285), [#286](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/286), [#287](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/287)).
- `osmosis train submit`, `eval submit`, and `benchmark submit` can supply per-run secrets from a dotenv file, standard input, the process environment, or an interactive prompt without saving them to the platform secret store ([#290](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/290)).
- `osmosis quickstart` now guides users through authentication, workspace repository setup, cloning, billing checks, and a ready-to-paste agent prompt, with matching workspace and quickstart APIs on `OsmosisClient` ([#299](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/299)).

### Fixed

- Hardened rollout packaging, finite LocalBackend deadlines, reward validation and callback delivery, Harbor credential and artifact handling, native-agent diagnostics, ATIF trajectory preservation, and secrets-file parsing ([#277](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/277), [#279](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/279), [#280](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/280), [#281](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/281), [#298](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/298)).

See [Migrating from 0.2.31 to 0.3](docs/migrating-to-0.3.md) for the complete upgrade checklist.

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.31...v0.3.0)

## 0.3.0rc4 - 2026-08-11

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.0rc4)

## 0.3.0rc3 - 2026-08-07

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.0rc3)

## 0.3.0rc2 - 2026-08-07

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.0rc2)

## 0.3.0rc1 - 2026-07-28

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.0rc1)

## 0.2.31 - 2026-07-28

### Changed

- LiteLLM 1.91.1 is now the minimum supported version for the SDK's model integrations ([#268](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/268)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.30...v0.2.31)
