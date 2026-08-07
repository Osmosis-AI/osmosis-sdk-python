# Changelog

This file records changes to `osmosis-ai`. For earlier versions, see [GitHub Releases](https://github.com/Osmosis-AI/osmosis-sdk-python/releases).

## Unreleased

### Breaking Changes

- Runtime features now use independent installation extras: `server`, `strands`, `openai-agents`, `harbor`, `rubric`, and `parquet`; `full` installs all of them. The former `platform` extra is replaced by `parquet`.
- Development tools are no longer published through the `dev` extra. From a source checkout, install the PEP 735 dependency group with `uv sync --all-extras --group dev` or `python -m pip install -e ".[full]" --group dev`.
- Rollout feature imports moved out of `osmosis_ai.rollout`:
  - Server: `from osmosis_ai.rollout.server import create_rollout_server, ControllerAuth`
  - Strands: `from osmosis_ai.rollout.integrations.agents.strands import OsmosisStrandsAgent, OsmosisRolloutModel`
  - OpenAI Agents: `from osmosis_ai.rollout.integrations.agents.openai_agents import OsmosisAgent`
  - Harbor: `from osmosis_ai.rollout.backend.harbor import HarborBackend, TaskMode`
- The legacy Harbor backend is gone and `HarborBackend` is now the implementation previously called `HarborBackendV2`, with a different constructor. `HarborAgentWorkflowContext` no longer exists — the workflow runs inside the container. See "Migrating from the pre-v0.3 Harbor backend" in [docs/rollout-sdk.md](docs/rollout-sdk.md).
- `evaluate_rubric` is no longer included by `from osmosis_ai import *`. Import it explicitly from `osmosis_ai.eval.rubric` and install the `rubric` extra.
- The Harbor extra no longer installs Daytona or SkyPilot. Daytona support is retired; the rollout runtime must provide SkyPilot when it is used.

### Added

- `AgentWorkflowOutput` is exported from `osmosis_ai.rollout`; it and `Messages` are exported from `osmosis_ai.rollout.types`. `AgentWorkflow.run` is now typed to return `AgentWorkflowOutput | Messages | None`: the return value carries the rollout's single message history in `messages` and is the primary trajectory source, while `None` falls back to the sample collected on the active `RolloutContext`. Unknown output fields (including the pre-v0.3 `samples` mapping) and non-finite metric values are rejected.
- `osmosis benchmark submit`, `osmosis eval submit`, and `osmosis train submit` accept `--secrets-file` to supply per-run values for `[secrets]` names without saving them to the secret store. Each name resolves by first hit: the dotenv file (`-` reads stdin), the process environment, the platform secret store, then an interactive prompt when stdin is a TTY. Outside a TTY all unresolved names are reported at once. See "Secret resolution" in [docs/eval.md](docs/eval.md).
- Benchmark configs accept `[verifier].required`, the secret record names a dataset's verifier reads, which submit forwards as `execution.verifier_secrets`. A name cannot also appear in `[env]` or an agent's `[agents.env]`, since the platform injects the secret value under that name. `osmosis benchmark info` reports `requires_judge_api_key` in JSON output.

## 0.3.0rc1 - 2026-07-28

### Breaking Changes

- Each rollout now produces one `RolloutSample` and one reward; migrate `samples` mappings to `sample`, `register_sample_source()` to `set_sample_source()`, and `set_sample_reward()` to `set_reward()` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Routing now uses rollout-scoped chat-completion and callback URLs; per-call routing headers were removed, request-body `rollout_id` is optional, and `RolloutSample.id` and `MultiTurnMode` no longer exist ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Harbor and local backends now exchange `sample.json` and write `reward.json` as `{"reward": <float>}` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.30...v0.3.0rc1)
