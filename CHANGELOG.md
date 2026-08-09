# Changelog

This file records changes to `osmosis-ai`. For earlier versions, see [GitHub Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases).

## Unreleased

### Fixed

- Rollout bundle builds now use content-addressed cache entries, keep project versions separate, normalize all valid distribution-name separators, and alias generated imports so same-named workflow, grader, and config objects cannot shadow one another. The Harbor extra now installs its `uv` builder directly, and model-backed extras stay below the incompatible LiteLLM 1.95 source-build boundary.

### Documentation

- Clarified that a Harbor task's existing `tests/test.sh` takes precedence over an SDK `grader=`, and that Harbor `TrialQueue` attempt retries are not supported by the rollout lifecycle integration.

## 0.3.0rc3 - 2026-08-07

### Breaking Changes

- Datasets must now use one uniform schema: prompt datasets require `user_prompt` and `ground_truth` (`label` remains an alias), while metadata datasets require a non-empty `metadata` object on every row; `system_prompt` is optional in both modes ([#282](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/282)).
- Benchmark detail consumers must replace `required_secret_names` with `requires_judge_api_key` in `BenchmarkCatalogDetail` and `osmosis benchmark info --json` output ([#290](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/290)).

### Added

- `osmosis train submit`, `eval submit`, and `benchmark submit` can supply per-run `[secrets]` values from `--secrets-file`, the process environment, or an interactive prompt without saving them to the platform secret store ([#290](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/290)).
- Benchmark configs now support `[verifier].required` stored secrets and `[secrets].required` per-run values, and agent environments may set `HF_TOKEN` ([#290](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/290)).

### Changed

- Dataset validation now scans every JSONL and CSV row plus every Parquet metadata value, rejecting inconsistent JSONL fields or invalid metadata anywhere in the file ([#282](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/282)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.0rc2...v0.3.0rc3)

## 0.3.0rc2 - 2026-08-07

### Breaking Changes

- Runtime features now use the `server`, `strands`, `openai-agents`, `harbor`, `rubric`, and `parquet` extras, with `full` aggregating them; replace the former `platform` extra with `parquet`, use the source dependency group instead of the published `dev` extra, and provide SkyPilot through the rollout runtime because Daytona support is retired ([#270](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/270)).
- Import server, Harbor, Strands, and OpenAI Agents features from their explicit submodules instead of `osmosis_ai.rollout`, and import `evaluate_rubric` explicitly from `osmosis_ai.eval.rubric` instead of relying on `from osmosis_ai import *` ([#270](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/270)).
- `AgentWorkflow.run()` now returns `AgentWorkflowOutput | Messages | None` for one message history; replace the pre-v0.3 `samples` mapping with `messages`, return `None` for ambient sample fallback, and ensure output metrics are finite ([#270](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/270)).
- The legacy `HarborBackend` is removed and its name now refers to the container-native implementation previously called `HarborBackendV2`; migrate constructor arguments and replace `HarborAgentWorkflowContext` with `AgentWorkflowContext` ([#291](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/291), [#292](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/292)).
- The removed `osmosis_ai.eval.common` loader helpers and `osmosis_ai.rollout.validator` module have no public drop-in replacement: rollout entrypoints now construct their backend explicitly, submit preflight imports the entrypoint without namespace discovery, and backend constructors own validation. `save_trajectories()` is replaced by the single-sample `save_trajectory()` API. See [Migrating from 0.2.31 to 0.3](docs/migrating-to-0.3.md) ([#270](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/270), [#277](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/277)).

### Added

- Added benchmark catalog, submission, run inspection, logs, cancellation, and output downloads through the `osmosis benchmark` CLI, with strict TOML validation and structured Rich/JSON output ([#265](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/265)).
- Added the container-native Harbor backend with installable workflow bundles, native Harbor agents, template and dataset task modes, prewarming, lifecycle diagnostics, artifact collection, and cancellation ([#272](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/272)).
- Rollout servers now return `202` for accepted work, reject full queues with `429` and `Retry-After`, and expose status polling and cancellation endpoints ([#287](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/287)).

### Fixed

- Hardened Harbor credential and artifact handling, surfaced agent failures and native-agent authentication errors, preserved callback outcomes and diagnostics, and retained complete samples, tool turns, rewards, and ATIF trajectories across container boundaries ([#277](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/277), [#279](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/279), [#280](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/280), [#281](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/281)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.0rc1...v0.3.0rc2)

## 0.3.0rc1 - 2026-07-28

### Breaking Changes

- Each rollout now produces one `RolloutSample` and one reward; migrate `samples` mappings to `sample`, `register_sample_source()` to `set_sample_source()`, and `set_sample_reward()` to `set_reward()` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Routing now uses rollout-scoped chat-completion and callback URLs; per-call routing headers were removed, request-body `rollout_id` is optional, and `RolloutSample.id` and `MultiTurnMode` no longer exist ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Harbor and local backends now exchange `sample.json` and write `reward.json` as `{"reward": <float>}` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.30...v0.3.0rc1)

## 0.2.31 - 2026-07-28

### Changed

- LiteLLM 1.91.1 is now the minimum supported version for the SDK's model integrations ([#268](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/268)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.30...v0.2.31)
