<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/osmosis-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset=".github/osmosis-logo-light.svg">
    <img alt="Osmosis" src=".github/osmosis-logo-light.svg" width="218">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue"></a>
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/osmosis-ai?color=yellow"></a>
  <a href="https://pypi.org/project/osmosis-ai/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/osmosis-ai"></a>
  <a href="https://codecov.io/gh/Osmosis-AI/osmosis-sdk-python"><img alt="Codecov" src="https://codecov.io/gh/Osmosis-AI/osmosis-sdk-python/branch/main/graph/badge.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-orange.svg"></a>
  <a href="https://docs.osmosis.ai"><img alt="Docs" src="https://img.shields.io/badge/docs-docs.osmosis.ai-green"></a>
</p>

# osmosis-ai

> ⚠️ **Warning**: osmosis-ai is still in active development. APIs may change between versions.

Python SDK for [Osmosis AI](https://platform.osmosis.ai), a platform for training LLMs with reinforcement learning. Implement an **AgentWorkflow** in Python, add a concrete **Grader** for local eval and managed training flows, validate the rollout entrypoint locally with the CLI, then let Osmosis managed hosting load it through Git Sync.

## Quick start

| Step | What you do |
|------|-------------|
| **Define agents** | One `AgentWorkflow` subclass (+ optional `AgentWorkflowConfig`) in your repo. The training/eval entrypoint must also expose a concrete `Grader` (typically with a `GraderConfig`). |
| **Layout** | Use a rollout pack directory under `rollouts/<name>/` when loading by rollout name; the CLI adds that directory to `sys.path`. |
| **Link project** | `osmosis project link --workspace <workspace-id-or-name>` — run from the workspace's connected Git Sync repo checkout so platform commands resolve the correct workspace. |
| **Validate** | `osmosis rollout validate configs/training/<name>.toml` — validate the rollout entrypoint referenced by a training or eval config before managed hosting or training submission. |
| **Evaluate** | `osmosis eval run configs/eval/<name>.toml` — same execution stack as training, with optional pass@k and caching. |

**Example repositories:** [osmosis-git-sync-example](https://github.com/Osmosis-AI/osmosis-git-sync-example) (synced agent repo patterns) · [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) (reference server usage — align with current SDK exports when upgrading).

**Documentation index:** [docs/README.md](docs/README.md)

## Agent-friendly CLI output

The `osmosis` CLI keeps Rich as the default output for humans, but every public command also speaks structured JSON and low-noise plain text for AI agents, CI/CD, and shell automation. Put these global output flags before the command:

```bash
osmosis dataset list                         # human-friendly Rich table
osmosis --json dataset list                  # recommended for AI agents and CI/CD
osmosis --format plain dataset list          # low-noise text for shell pipelines
```

JSON is the stable machine contract: every successful response includes `schema_version: 1`; list envelopes include `items`, `total_count`, `has_more`, and `next_offset`; detail envelopes include `data`; operation envelopes include `status`, `operation`, optional `resource`, and optional `next_steps_structured`. Errors are JSON-structured on stderr with `code`, `message`, `details`, optional `request_id`, plus the command path and SDK `cli_version`.

Plain mode is for humans and simple shell pipelines, not a strict schema. `--format` and its `--json` / `--plain` aliases are global flags parsed before subcommands; prefer `osmosis --json <command>` or `osmosis --format plain <command>` over the default Rich output in non-interactive environments. Command-local `--output` always means a file path, not a format selector, so `osmosis dataset download my-dataset --output ./data.jsonl` works in every mode.

In JSON or plain mode, interactive commands fail fast with `INTERACTIVE_REQUIRED` unless a non-interactive flow exists, typically by passing `--yes` or `--token`. `OSMOSIS_TOKEN` is verify-only across the CLI: it activates authentication for the current process but is never written to the on-disk credentials store, never revoked, and never deletes existing credentials.

## Project creation and CI

Create an Osmosis project in a new directory, or use `--here` from a completely
empty current directory:

```bash
osmosis init <name>
# or, from an empty directory:
osmosis init --here <name>
osmosis project link --workspace <workspace-id-or-name>
```

Submit from CI with a project link and non-interactive confirmation:

```bash
export OSMOSIS_TOKEN=<token>
osmosis project link --workspace <workspace-id-or-name> --yes
osmosis train submit configs/training/default.toml --yes
```

## Installation

Requires **Python 3.12+**. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

- **An LLM API key** (e.g., OpenAI, Anthropic, Groq) — required for `osmosis eval run` when using hosted models. See [supported providers](https://docs.litellm.ai/docs/providers).
- **Osmosis account** (optional) — needed for `osmosis auth login`, project links, and platform-backed commands such as datasets, models, and training runs. Sign up at [platform.osmosis.ai](https://platform.osmosis.ai).

**pip**

```bash
pip install osmosis-ai            # Core SDK
pip install osmosis-ai[server]    # + FastAPI rollout server (pulls in platform extra)
pip install osmosis-ai[full]      # Same as [server] (all packaged optional features)
```

**uv**

```bash
uv add osmosis-ai                 # Core SDK
uv add osmosis-ai[server]         # + FastAPI rollout server (pulls in platform extra)
uv add osmosis-ai[full]           # Same as [server] (all packaged optional features)
```

## Testing and evaluation

- [Eval](docs/eval.md) — graded runs, pass@k, cache/resume with `osmosis eval run`
- [CLI reference](docs/cli.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, linting, and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
