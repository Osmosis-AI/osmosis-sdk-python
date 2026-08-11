# Changelog

This file records changes to `osmosis-ai`. For earlier versions, see [GitHub Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases).

## Unreleased

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
