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
| Eval helpers | [../osmosis_ai/eval/](../osmosis_ai/eval/) | Rubric (LLM-as-judge) + workflow/grader loader | `from osmosis_ai.eval.rubric import evaluate_rubric` |
| Workspace templates | [../osmosis_ai/templates/](../osmosis_ai/templates/) | `osmosis template` recipe catalog + source resolution | (internal) |

## Pages

- [architecture.md](./architecture.md) — package layout, domain boundaries, import paths, lazy-loading rules, and the remote rollout protocol (controller <-> rollout server). Start here.
- [rollout-sdk.md](./rollout-sdk.md) — the library API you implement against: `AgentWorkflow`, `Grader`, contexts, configs, `create_rollout_server`, execution backends, and framework integrations.
- [eval.md](./eval.md) — the `osmosis eval submit` config contract (SDK-vs-backend validation, submit flow), plus a brief note on the `evaluate_rubric` / `osmosis eval rubric` LLM-as-judge API.
- [datasets.md](./datasets.md) — the dataset row contract enforced by the SDK validator.
- [troubleshooting.md](./troubleshooting.md) — engineering issues (rollout timeouts, event-loop blocking, concurrency tuning).
- [cli.md](./cli.md) — CLI internals for contributors (command shells, lazy imports, JSON envelopes).

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev environment, tests, lint, type checking, PR conventions
- [docs.osmosis.ai/cli/command-reference](https://docs.osmosis.ai/cli/command-reference) — the user-facing command reference
