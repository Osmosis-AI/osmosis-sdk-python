# Changelog

This file records changes to `osmosis-ai`. For earlier versions, see [GitHub Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases).

## 0.3.5rc1 - 2026-09-24

### Breaking Changes

- The internal `osmosis dev server up|down|list|logs` commands are removed from the public SDK; internal users run the same commands from the private `osmo` CLI as `osmo dev server ...` ([#389](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/389)).
- The dedicated dev-server Python API is removed: the `OsmosisClient` methods `provision_dev_rollout_server`, `teardown_dev_rollout_server`, `get_dev_rollout_server_logs`, `stream_dev_rollout_server_logs`, and `list_dev_rollout_servers`, the `DevRolloutServerInfo` and `PaginatedDevRolloutServers` models, the `DevServerSandboxEnvironment` and `DevServerBackend` enums, and `serialize_dev_rollout_server` ([#389](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/389)).

### Added

- `osmosis images build --url URL --path PATH --ref REF` builds Harbor task images from a pinned GitHub task directory or Harbor Hub `dataset.toml` without downloading a task bundle, and `--output` writes the verified task-to-image JSON mapping ([#387](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/387), [#390](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/390)).
- `HarborGatewayConfig` in `osmosis_ai.rollout.types.harbor` validates the native Harbor agent, concurrency, environment, and healthcheck settings used by managed source gateways ([#390](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/390)).
- `osmosis_ai.cli.main.create_app()` and `run_cli()` let downstream CLIs reuse the public command tree, root options, and output handling while adding their own commands, version reporting, and upgrade command ([#389](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/389)).

### Changed

- JSON error envelopes resolve `command` from the invoked command tree: unknown subcommands are kept (for example `dataset lst`), `--workspace` values are no longer reported as the command, and `osmosis_ai.cli.output.command_path_for_error()` without `root_command` no longer reads `sys.argv` ([#389](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/389)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.4...v0.3.5rc1)

## 0.3.4 - 2026-09-23

### Added

- `osmosis images build --repo URL` builds connected job repositories and downloads verified task bundles with resumable requests; `osmosis images info` and Python client methods expose progress and artifacts ([#378](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/378)).
- `osmosis eval retry <name|id>` retries failed and skipped hosted samples while preserving graded results; local retries can replace uploaded results with `eval upload --replace` or `eval run --retry-failed --upload` ([#368](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/368)).
- `osmosis model info` and `OsmosisClient.get_base_model()` accept base model names or Hugging Face paths and expose parameter counts, context windows, and available inference pricing ([#370](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/370)).
- `osmosis dev server up` accepts `--backend ecs|gke` and `--sandbox-environment daytona|opensandbox` to select deployment placement and sandbox providers independently ([#371](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/371)).
- `HarborBackend(agent="opencode")` runs native OpenCode through the rollout chat-completions endpoint with compaction and pruning disabled; native agent options, including version pins, also apply during prewarm ([#374](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/374)).
- Rollout servers can export ownership and lifecycle logs over OTLP/HTTP by setting `OSMOSIS_ROLLOUT_OTLP_ENDPOINT`, with server and optional training-run attribution ([#375](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/375)).

### Changed

- The SDK now requires `anyio>=4.14.2` for security fixes; update older AnyIO pins when upgrading ([#384](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/384)).

### Fixed

- Harbor artifact archiving no longer blocks the event loop or stalls result polling on slow storage; rollouts still wait for archiving to finish before returning results ([#386](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/386)).
- `RolloutClient` retries transient result-polling disconnects and read timeouts up to twice within the last confirmed polling lease deadline; remove custom result-GET retry wrappers to avoid nested retries ([#366](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/366), [#374](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/374)).
- Identifiable sandbox provider failures now report `HTTP_ERROR`; task-level API errors and generic Harbor API errors without structured status remain `AGENT_ERROR` ([#373](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/373)).
- CLI failures now show actionable platform and login explanations with consistent HTTP error categories, and skipped rollout preflight warnings include installation guidance for the CLI's environment ([#369](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/369), [#376](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/376)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.3...v0.3.4)

## 0.3.4rc3 - 2026-09-23

See the [0.3.4rc3 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.4rc3).

## 0.3.4rc2 - 2026-09-21

See the [0.3.4rc2 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.4rc2).

## 0.3.4rc1 - 2026-09-15

See the [0.3.4rc1 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.4rc1).

## 0.3.3 - 2026-09-14

### Breaking Changes

- Rollout completion now uses leased long polling: upgrade callers and servers together, replace `HttpRolloutDriver` and callback models with `RolloutClient`, and supply a unique `rollout_id` plus `llm_api_key` when the chat endpoint requires authentication ([#347](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/347)).
- Custom backends must return `ExecutionOutcome` from `execute(request)` and publish progress with `await RolloutContext.set_status()`; move local LLM bridge imports from `osmosis_ai.rollout.controller` to `osmosis_ai.eval.local` ([#347](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/347), [#355](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/355)).
- Local eval's protocol fingerprint is now `0.4`, so runs recorded with the previous protocol require a new run name or the previous SDK to resume ([#347](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/347)).
- Managed SkyPilot placement is removed: migrate to `EnvironmentConfig(type=EnvironmentType.DAYTONA)` with Daytona credentials; `HARBOR_SKYPILOT_CONTEXT` is no longer read and the `harbor` extra no longer installs `dockerfile-parse` ([#362](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/362)).

### Added

- `RolloutClient` handles lease renewal, HTTP 429 retries, cancellation, and optional `grade=False`; `await run_rollout_async()` returns an awaitable `RolloutHandle` with live status and `wait_for_running()`, `wait_for_grading()`, and `wait_for_completion()` methods ([#346](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/346), [#354](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/354), [#355](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/355)).
- `osmosis eval upload` and `osmosis eval run --upload` include `logs.txt` in the platform Logs tab, with known ambient provider and platform credentials of at least eight characters redacted during logging and again before upload ([#353](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/353)).

### Changed

- Built-in Daytona environments with `delete=True` default to stopping after 60 minutes of Daytona-observed inactivity and immediate deletion; increase `environment_config.kwargs["auto_stop_interval_mins"]` for longer idle periods or set it to `0` to disable ([#361](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/361)).
- Expanded support to Harbor `>=0.20.0,<0.23` and OpenAI Agents `>=0.18.1,<0.21`; when upgrading Harbor, remove top-level `trajectory.json` inputs from SDK workflow tasks and ensure Docker hosts pass the nftables probe for restricted-network trials ([#344](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/344)).

### Fixed

- Polling preserves completed rewards and failure details, applies admission deadlines to HTTP requests and retries, bounds cancellation cleanup, and leaves local eval work without a recorded terminal result pending for resume ([#348](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/348), [#349](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/349), [#351](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/351)).
- Login falls back to an owner-only credentials file when the system keyring is unavailable; set `OSMOSIS_TOKEN_STORE=keyring` to require the keyring or `file` to select file storage explicitly ([#364](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/364)).
- `osmosis train info` lists checkpoints for running training runs ([#360](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/360)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.2...v0.3.3)

## 0.3.3rc3 - 2026-09-11

See the [0.3.3rc3 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.3rc3).

## 0.3.3rc2 - 2026-09-09

See the [0.3.3rc2 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.3rc2).

## 0.3.3rc1 - 2026-09-07

See the [0.3.3rc1 release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.3rc1).

## 0.3.2 - 2026-08-31

### Breaking Changes

- Removed the public `MessageResult`, `GraderInitRequest`, `GraderInitResponse`, `RolloutDriver`, and `resolve_workspace_directory_from_cwd()` APIs; replace `MessageResult` with `OperationResult`, `RolloutDriver` with `HttpRolloutDriver`, and workspace lookup with `resolve_workspace_directory()` ([#328](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/328)).
- The SDK no longer declares `requests` as a base dependency or `tqdm` in the `rubric` extra; declare either package directly if your project imports it ([#328](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/328)).

### Added

- `osmosis eval run` now reaches Daytona, SkyPilot, and other cloud sandboxes through a managed `cloudflared` or bring-your-own tunnel, started automatically when the sandbox cannot reach host loopback ([#327](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/327), [#335](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/335)).
- Added a root `--workspace <name>` selector for workspace-scoped platform commands; benchmark operations can now run without a local repository, while training and evaluation submit accept an absolute config path when its Git repository matches the selected workspace ([#332](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/332)).
- `osmosis eval upload <run-name>` now resolves completed runs from `.osmosis/evals`, while explicit custom directories remain supported ([#335](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/335)).
- Documented the `eval` installation extra and declared Python 3.14 support ([#337](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/337)).

### Changed

- Local `osmosis eval run --dataset-file ...` no longer loads platform credentials unless `--upload` is also requested ([#332](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/332)).
- Local eval output, retry, resume, and upload paths are now shown relative to the invocation directory when possible, keeping generated commands copyable from workspace subdirectories ([#335](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/335)).
- Newly scaffolded rollouts now depend on the stable `osmosis-ai[server]>=0.3.0,<0.4` release line instead of an RC baseline ([#337](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/337)).

### Fixed

- Hardened local eval startup and tunneling: the model and rollout server are validated before a tunnel opens, slow non-streaming responses stay alive, a registered Cloudflare connection is accepted when the host cannot probe the tunnel URL, and unreachable endpoints fail fast instead of hanging ([#327](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/327), [#335](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/335), [#337](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/337)).
- Local evaluation now warns when the CLI and rollout environments use different `osmosis-ai` versions ([#330](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/330)).
- The `harbor` extra now installs Harbor's Daytona dependencies so Daytona environments work without separate dependency setup ([#331](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/331)).
- Secrets-file values now override the process environment only for the local eval run and are restored on every exit path, and orphan cleanup rejects symlinked run directories and state files ([#337](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/337)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.3.1...v0.3.2)

## 0.3.2rc3 - 2026-08-31

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.2rc3)

## 0.3.2rc2 - 2026-08-31

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.2rc2)

## 0.3.2rc1 - 2026-08-28

[Incremental release notes](https://github.com/Osmosis-AI/osmosis-sdk-python/releases/tag/v0.3.2rc1)

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
