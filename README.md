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

Python SDK for [Osmosis AI](https://platform.osmosis.ai), a platform for training LLMs with reinforcement learning. Implement an **AgentWorkflow** in Python, add a concrete **Grader** for local eval and managed training flows, run an eval smoke test locally with the CLI, then submit training from an Osmosis workspace directory.

## Quick start

| Step | What you do |
|------|-------------|
| **Define agents** | One `AgentWorkflow` subclass (+ optional `AgentWorkflowConfig`) in your repo. The training/eval entrypoint must also expose a concrete `Grader` (typically with a `GraderConfig`). |
| **Layout** | Use a rollout pack directory under `rollouts/<name>/` when loading by rollout name; the CLI adds that directory to `sys.path`. |
| **Workspace directory** | Create or open a workspace in the Osmosis Platform, then clone the repository created there. |
| **Check workspace** | `osmosis workspace doctor` — run from the workspace directory so platform commands resolve the repository from the `origin` remote. Add `--fix` to restore missing scaffold paths. |
| **Smoke test** | `osmosis eval run configs/eval/<name>.toml --limit 1` — exercises the same rollout server protocol used by training. |
| **Evaluate** | `osmosis eval run configs/eval/<name>.toml` — run the full eval with optional pass@k and caching. |

**Example repository:** [osmosis-remote-rollout-example](https://github.com/Osmosis-AI/osmosis-remote-rollout-example) (reference server usage - align with current SDK exports when upgrading).

**Documentation index:** [docs/README.md](docs/README.md)

## Agent-friendly CLI output

The `osmosis` CLI keeps Rich as the default output for humans, but every public command also speaks structured JSON and low-noise plain text for AI agents, CI/CD, and shell automation. Put these global output flags before the command:

```bash
osmosis dataset list                         # human-friendly Rich table
osmosis --json dataset list                  # recommended for AI agents and CI/CD
osmosis --plain dataset list                 # low-noise text for shell pipelines
```

JSON is the stable machine contract: every successful response includes `schema_version: 1`; list envelopes include `items`, `total_count`, `has_more`, and `next_offset`; detail envelopes include `data`; operation envelopes include `status`, `operation`, optional `resource`, and optional `next_steps_structured`. Errors are JSON-structured on stderr with `code`, `message`, `details`, optional `request_id`, plus the command path and SDK `cli_version`.

Plain mode is for humans and simple shell pipelines, not a strict schema. `--json` and `--plain` are global flags parsed before subcommands; prefer `osmosis --json <command>` or `osmosis --plain <command>` over the default Rich output in non-interactive environments. Command-local `--output` always means a file path, not a format selector, so `osmosis dataset download my-dataset --output ./data.jsonl` works in every mode.

In JSON or plain mode, interactive commands fail fast with `INTERACTIVE_REQUIRED` unless a non-interactive flow exists, typically by passing `--yes` or `--token`. `OSMOSIS_TOKEN` is verify-only across the CLI: it activates authentication for the current process but is never written to the on-disk credentials store, never revoked, and never deletes existing credentials.

## Workspace Directory Flow

Create or open a workspace in the Osmosis Platform, clone the repository created there,
then run CLI commands from that workspace directory.

```bash
git clone <repo-url>
cd <repo>
osmosis auth login
osmosis workspace doctor
osmosis template apply multiply              # or add your rollout under rollouts/
cp configs/training/default.toml configs/training/<run>.toml
$EDITOR configs/training/<run>.toml          # set rollout, dataset, and model_path
git add rollouts configs data research
git commit -m "configure training run"
git push
osmosis train submit configs/training/<run>.toml
```

Platform-scoped commands derive scope from the workspace directory's `origin` remote and
send `X-Osmosis-Git: namespace/repo_name`. The CLI does not store or send a
workspace ID for commands scoped by the workspace directory.

Before submitting training from CI, push the repository and authenticate with a
token:

```bash
export OSMOSIS_TOKEN=<token>
osmosis train submit configs/training/<run>.toml --yes
```

## Installation

Requires **Python 3.12+**. For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

- **An LLM API key** (e.g., OpenAI, Anthropic, Groq) — required for `osmosis eval run` when using hosted models. See [supported providers](https://docs.litellm.ai/docs/providers).
- **Osmosis account** (optional) — needed for `osmosis auth login` and platform-backed commands such as datasets, models, and training runs. Sign up at [platform.osmosis.ai](https://platform.osmosis.ai).

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
