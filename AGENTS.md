# AGENTS.md

Repository guidance for AI agents and developers. Authoritative reference material lives in the linked docs.

## What this repo is

`osmosis-ai` is the Python SDK and CLI for [Osmosis AI](https://platform.osmosis.ai). Users implement an `AgentWorkflow` and a concrete `Grader`, then submit evaluation and training runs with `osmosis`. Package metadata and console entry points live in [pyproject.toml](pyproject.toml).

## Choose the task entry point

| Task | Read |
|------|------|
| Locate implementation, tests, or a contract | [Task map](docs/README.md#task-map); follow the relevant row |
| Change rollout execution or review compatibility | [Architecture](docs/architecture.md#runtime-boundaries), then the relevant contract |
| Change dependencies or extras | [Dependency changes](CONTRIBUTING.md#dependency-changes) |
| Run checks, prepare a PR, or release | [Verification](CONTRIBUTING.md#verification) and [PR conventions](CONTRIBUTING.md#pull-requests) |
| Create a pull request | Use the [create-pr skill](.agents/skills/create-pr/SKILL.md) |
| Explain installation or user-facing usage | [README.md](README.md) and [product docs](https://docs.osmosis.ai) |

## Documentation

When code and a doc disagree, update the doc with the code change. Keep prose paragraphs and individual list items on one line unless Markdown structure requires a break. Add developer reference pages to [docs/README.md](docs/README.md); link to source files so the existing documentation checks can detect moved targets.
