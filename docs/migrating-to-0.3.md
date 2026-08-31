# Migrating from 0.2.31 to 0.3

Version 0.3 intentionally breaks the pre-0.3 rollout contract. This page is the integrated migration checklist for users upgrading from the latest 0.2 release; the RC-by-RC history remains in [CHANGELOG.md](../CHANGELOG.md).

## Installation and imports

The base distribution now contains the CLI and framework-neutral rollout core. Install only the extras used by the rollout: `server`, `eval`, `strands`, `openai-agents`, `harbor`, `rubric`, or `parquet`; `full` installs every optional feature. The former `platform` extra and published `dev` extra are gone, and SkyPilot is supplied by the managed rollout runtime rather than Harbor's conflicting `skypilot` extra.

Server, Harbor, Strands, and OpenAI Agents integrations must be imported from their explicit submodules. `LocalBackend`, `AgentWorkflow`, `Grader`, and their core types remain available from `osmosis_ai.rollout`; import `HarborBackend` from `osmosis_ai.rollout.backend.harbor` and `evaluate_rubric` from `osmosis_ai.eval.rubric`.

## One rollout, one sample

Each rollout now produces one `RolloutSample` and one reward. Replace `samples` mappings with singular `sample`, `register_sample_source()` with `set_sample_source()`, and `set_sample_reward()` with `set_reward()`. `RolloutSample.id` and `MultiTurnMode` no longer exist. `AgentWorkflow.run()` returns `AgentWorkflowOutput | Messages | None`: return one message history, or `None` to use the ambient sample populated through `RolloutContext`.

Routing identity now lives in the rollout-scoped chat-completion and callback URLs. Rollout servers send singular `sample` callback payloads, which controllers must accept; agents must not rely on the removed per-call `x-rollout-id` or `x-sample-id` headers.

## Removed loader and validator APIs

The following public APIs were removed rather than deprecated because 0.3 is the compatibility boundary:

| Pre-0.3 API | 0.3 migration |
|-------------|---------------|
| `osmosis_ai.eval.common.load_workflow` | Construct the backend and rollout server explicitly in the configured entrypoint. The platform submit preflight imports that module; there is no public discovery helper. |
| `osmosis_ai.eval.common.auto_discover_grader` | Pass the grader explicitly to `LocalBackend(grader=...)` or `HarborBackend(grader=...)`. |
| `osmosis_ai.rollout.validator.ValidationError` / `ValidationResult` / `resolved_agent_name` / `validate_backend` | Remove the static validator call. Backend constructors validate their own configuration, import-time failures are reported by submit preflight, and an eval run is the end-to-end smoke test before training. |
| `osmosis_ai.rollout.trajectory.save_trajectories` | Use `await save_trajectory(...)`; `ExecutionResult` now contains at most one sample, and the output is `<artifact_root>/<rollout_id>/trajectory.json`. |

Submit preflight no longer scans an entrypoint namespace for workflow or grader subclasses. It imports the configured entrypoint once so module-level backend construction can validate itself; errors that require a real model, task, or container surface on the first rollout.

## Harbor backend

`HarborBackend` now names the container-native implementation that was called `HarborBackendV2`. It builds an installable wheel from the rollout project rather than mounting the SDK and source tree into a task. See the constructor mapping and reward precedence rules in [rollout-sdk.md](./rollout-sdk.md#migrating-from-the-pre-v03-harbor-backend).

Harbor task verifiers remain authoritative: if a task already contains `tests/test.sh`, it is used even when `grader=` is supplied. To run the SDK grader, use a task without that file. Keep `TrialQueue` at its default `RetryConfig(max_retries=0)` because the SDK treats Harbor's `END` event as terminal.

## Recommended upgrade check

After updating imports and constructing the backend explicitly, run one evaluation with the same rollout entrypoint, task source, agent integration, and grader intended for training. This exercises dependency installation, entrypoint import, model routing, container packaging, callback delivery, and reward production together.
