# Osmosis SDK developer docs

> Product concepts and end-user CLI usage live at **[docs.osmosis.ai](https://docs.osmosis.ai)**. This `docs/` directory is the **code-anchored** reference for developers building on the SDK. Every page points at the source it documents; when code and a doc disagree, the code wins — fix the doc in the same PR.

## Who reads what

| Audience | Home | Orientation |
|----------|------|-------------|
| End users / everyone | [docs.osmosis.ai](https://docs.osmosis.ai) | Platform concepts, onboarding, CLI usage, quickstart |
| SDK developers in this repo | this `docs/` directory | Importable APIs, contracts, architecture, behavior — anchored to source |

We accept small, deliberate duplication only for entry facts (e.g. a one-line install). For everything else there is one source of truth: usage and product concepts link out to the site; code contracts live here next to the code.

## Package map

The package (`osmosis_ai/`) is organized into top-level domains. See [architecture.md](./architecture.md) for the full layout and the rollout protocol.

| Domain | Source | Purpose | Primary import |
|--------|--------|---------|----------------|
| CLI framework + commands | [../osmosis_ai/cli/](../osmosis_ai/cli/) | Typer entry point, command shells, output/JSON envelopes | `from osmosis_ai.cli.errors import CLIError` |
| Platform integration | [../osmosis_ai/platform/](../osmosis_ai/platform/) | Auth, platform API client, CLI business logic | `from osmosis_ai.platform.auth import load_credentials` |
| Remote rollout SDK | [../osmosis_ai/rollout/](../osmosis_ai/rollout/) | `AgentWorkflow` + `Grader`, contexts, server, backends | `from osmosis_ai.rollout import AgentWorkflow, Grader` |
| Eval helpers | [../osmosis_ai/eval/](../osmosis_ai/eval/) | Rubric (LLM-as-judge) | `from osmosis_ai.eval.rubric import evaluate_rubric` |
| Workspace templates | [../osmosis_ai/templates/](../osmosis_ai/templates/) | `osmosis template` recipe catalog + source resolution | (internal) |

The single `osmosis-ai` distribution always includes the CLI and framework-neutral rollout core. Install extras only for the feature you use: `server`, `strands`, `openai-agents`, `harbor`, `rubric`, `parquet`, `eval`, or `full`. The Harbor extra installs Harbor with its Daytona environment dependencies.

## Task map

Start with the row matching the behavior you need to change or review. Tests do not exactly mirror package paths: authentication tests live in `tests/unit/auth/`, CLI tests span `tests/unit/cli/` and `tests/unit/platform/cli/`, and image/import contracts live directly under `tests/unit/`. Confirm narrower paths with `rg --files` before reading them.

| Task | Implementation entry | Focused tests | Reference |
|------|----------------------|---------------|-----------|
| Authentication and credential storage | [platform/auth/credentials.py](../osmosis_ai/platform/auth/credentials.py) | [tests/unit/auth/](../tests/unit/auth/) | [CLI internals](cli.md) |
| CLI commands and cloud submit preflight | [cli/commands/](../osmosis_ai/cli/commands/), [platform/cli/rollout_entrypoint.py](../osmosis_ai/platform/cli/rollout_entrypoint.py) | [tests/unit/cli/](../tests/unit/cli/), [tests/unit/platform/cli/](../tests/unit/platform/cli/) | [CLI internals](cli.md), [eval submit](eval.md) |
| Rollout admission, polling, cancellation, and terminal results | [server/app.py](../osmosis_ai/rollout/server/app.py), [server/result_registry.py](../osmosis_ai/rollout/server/result_registry.py), [client/client.py](../osmosis_ai/rollout/client/client.py) | [test_result_registry.py](../tests/unit/rollout/test_result_registry.py), [test_server_lifecycle.py](../tests/unit/rollout/test_server_lifecycle.py), [client tests](../tests/unit/rollout/client/test_client.py) | [Runtime boundaries](architecture.md#runtime-boundaries), [lifecycle](rollout-lifecycle.md) |
| Local workflow and grader execution | [backend/local/backend.py](../osmosis_ai/rollout/backend/local/backend.py), [container/runner.py](../osmosis_ai/rollout/container/runner.py) | [test_local_backend.py](../tests/unit/rollout/test_local_backend.py), [test_container_runner.py](../tests/unit/rollout/test_container_runner.py) | [Rollout SDK](rollout-sdk.md) |
| Trajectory persistence and artifact paths | [trajectory/save.py](../osmosis_ai/rollout/trajectory/save.py), [utils/file_artifacts.py](../osmosis_ai/rollout/utils/file_artifacts.py) | [test_trajectory_save.py](../tests/unit/rollout/test_trajectory_save.py), [test_file_artifacts.py](../tests/unit/rollout/test_file_artifacts.py), [test_server_app_trajectory.py](../tests/unit/rollout/test_server_app_trajectory.py) | [Rollout SDK](rollout-sdk.md) |
| Harbor execution, native agents, and evidence | [backend/harbor/backend.py](../osmosis_ai/rollout/backend/harbor/backend.py), [native_agents.py](../osmosis_ai/rollout/backend/harbor/native_agents.py), [evidence.py](../osmosis_ai/rollout/backend/harbor/evidence.py) | [test_harbor_backend.py](../tests/unit/rollout/test_harbor_backend.py), [test_harbor_trial_logs.py](../tests/unit/rollout/test_harbor_trial_logs.py), [test_native_evidence.py](../tests/unit/rollout/test_native_evidence.py) | [Rollout SDK](rollout-sdk.md), [runtime boundaries](architecture.md#runtime-boundaries) |
| Source-image identity and gateway binding | [harbor_images.py](../osmosis_ai/harbor_images.py), [source_images.py](../osmosis_ai/source_images.py), [backend/harbor/source.py](../osmosis_ai/rollout/backend/harbor/source.py) | [test_harbor_images.py](../tests/unit/test_harbor_images.py), [test_source_images.py](../tests/unit/test_source_images.py), [test_harbor_source_images.py](../tests/unit/rollout/test_harbor_source_images.py) | [Lifecycle and image types](rollout-lifecycle.md) |
| Rollout ownership/status telemetry | [server/observability.py](../osmosis_ai/rollout/server/observability.py), [server/app.py](../osmosis_ai/rollout/server/app.py) | [test_server_observability.py](../tests/unit/rollout/test_server_observability.py) | [Ownership logs](rollout-observability.md) |
| Optional dependencies, lazy imports, and wheel metadata | [pyproject.toml](../pyproject.toml), [_imports.py](../osmosis_ai/_imports.py), [verify-wheel-install.py](../.github/scripts/verify-wheel-install.py) | [test_public_api_imports.py](../tests/unit/test_public_api_imports.py), [wheel-smoke CI](../.github/workflows/tests.yml) | [Dependency changes](../CONTRIBUTING.md#dependency-changes) |

## Pages

- [architecture.md](./architecture.md) — package layout, domain boundaries, import paths, lazy-loading rules, and the remote rollout protocol (client <-> rollout server). Start here.
- [rollout-sdk.md](./rollout-sdk.md) — the library API you implement against: `AgentWorkflow`, `Grader`, contexts, configs, server/backends, and framework integrations.
- [rollout-lifecycle.md](./rollout-lifecycle.md) — authenticated health, admission drain, native evidence inventories, and source image types.
- [rollout-observability.md](./rollout-observability.md) — opt-in OTLP ownership/status logs, identity metadata, and exporter behavior.
- [migrating-to-0.3.md](./migrating-to-0.3.md) — the source and behavior changes required when upgrading an SDK integration from 0.2.31.
- [eval.md](./eval.md) — the `osmosis eval submit` config contract (SDK-vs-backend validation, submit flow), plus a brief note on the `evaluate_rubric` / `osmosis eval rubric` LLM-as-judge API.
- [eval-run-local.md](./eval-run-local.md) — `osmosis eval run`: local evaluation against your own rollout server, the run-directory layout, resume/fresh/retry semantics, secrets, and Harbor sandbox specifics.
- [benchmark.md](./benchmark.md) — the `osmosis benchmark submit` config contract: sections, the agent model union, secret-reference and env-collision rules, and the submit flow.
- [datasets.md](./datasets.md) — the dataset row contract enforced by the SDK validator.
- [troubleshooting.md](./troubleshooting.md) — engineering issues (rollout timeouts, event-loop blocking, concurrency tuning).
- [cli.md](./cli.md) — CLI internals for contributors (command shells, lazy imports, JSON envelopes).
- [run-downloads.md](./run-downloads.md) — eval and benchmark download commands, platform route contracts, fixed local layouts, resume, confirmation, and retry behavior.

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev environment, tests, lint, type checking, PR conventions
- [CHANGELOG.md](../CHANGELOG.md) — SDK changes by release
- [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference) — the user-facing command reference
