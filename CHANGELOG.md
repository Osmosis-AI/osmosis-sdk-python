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

## 0.3.0rc1 - 2026-07-28

### Breaking Changes

- Each rollout now produces one `RolloutSample` and one reward; migrate `samples` mappings to `sample`, `register_sample_source()` to `set_sample_source()`, and `set_sample_reward()` to `set_reward()` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Routing now uses rollout-scoped chat-completion and callback URLs; per-call routing headers were removed, request-body `rollout_id` is optional, and `RolloutSample.id` and `MultiTurnMode` no longer exist ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).
- Harbor and local backends now exchange `sample.json` and write `reward.json` as `{"reward": <float>}` ([#235](https://github.com/Osmosis-AI/osmosis-sdk-python/pull/235)).

[Full changelog](https://github.com/Osmosis-AI/osmosis-sdk-python/compare/v0.2.30...v0.3.0rc1)
