# Contributing

## Quick Start

**Using uv (recommended):**

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python
uv sync --extra dev
pre-commit install
uv run pytest
```

**Using pip:**

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Commands Reference

The table below lists all development commands. If you installed with **pip**, drop the `uv run` prefix.

| Task | Command |
|------|---------|
| Run all tests | `uv run pytest` |
| Run a single file | `uv run pytest tests/unit/rollout/core/test_base.py` |
| Run tests by name | `uv run pytest -k "test_name"` |
| Run with coverage | `uv run pytest --cov=osmosis_ai --cov-report=term-missing` |
| Lint | `uv run ruff check .` |
| Lint + autofix | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Check formatting | `uv run ruff format --check .` |
| Type check (pyright) | `uv run pyright osmosis_ai/` |
| Type check (mypy) | `uv run mypy osmosis_ai/` |

## Testing

Coverage configuration is in `pyproject.toml` under `[tool.coverage.*]`. CI enforces a minimum coverage threshold of 70%.

## Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and code formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

Ruff is pinned to one version across `pyproject.toml`, `.pre-commit-config.yaml`, and CI so local checks, pre-commit hooks, and GitHub Actions produce the same results.

## Type Checking

[Pyright](https://microsoft.github.io/pyright/) is the primary type checker and [mypy](https://mypy-lang.org/) is a secondary checker. Both are included in the `dev` extras.

- **Pyright** — must pass. All errors must be resolved before merging.
- **mypy** — advisory (`continue-on-error` in CI). Fix warnings when practical, but they won't block a PR.

Configuration for both tools lives in `pyproject.toml` under `[tool.pyright]` and `[tool.mypy]`.

> **Note:** CI also runs `pyright --verifytypes osmosis_ai --ignoreexternal` to check public API type completeness. This requires a non-editable install and must pass before merging.

## Pre-commit Hooks

[Pre-commit](https://pre-commit.com/) runs `ruff check --fix` and `ruff format` automatically on every commit. Make sure hooks are installed before submitting a pull request:

```bash
pre-commit install
```

> **Tip:** Run `uv run pyright osmosis_ai/` before pushing to catch type errors early. CI will block PRs with pyright failures.

## Pull Requests

### PR Title Format

All PR titles **must** follow this format (enforced by CI):

```
[module] type: description
[mod1][mod2] type: description          # multi-module (up to 3)
```

- **Modules**: `reward`, `rollout`, `server`, `cli`, `auth`, `eval`, `misc`, `ci`, `doc`
- **Types**: `feat`, `fix`, `refactor`, `chore`, `test`, `doc`

For breaking changes, add `[BREAKING]` before the modules:

```
[BREAKING][module] type: description
```

**Examples:**

```
[rollout] feat: add streaming support for chat completions
[rollout][auth] refactor: extract LifecycleManager
[server] fix: handle timeout in rollout init
[cli] chore: update dependency versions
[BREAKING][reward] refactor: rename decorator parameters
```

PR titles appear directly in auto-generated GitHub Release Notes, so keep them clear and descriptive.

### Labels

Add a label to your PR so it gets categorized correctly in Release Notes:

| Label | Use when |
|-------|----------|
| `enhancement` | New feature |
| `bug` | Bug fix |
| `breaking` | Breaking change |
| `documentation` | Docs update |
| `chore` / `ci` / `refactor` / `dependencies` | Maintenance work |

Module-specific labels (`reward`, `rollout`, `server`, `cli`, `auth`, `eval`) can also be added for filtering.

### Workflow

1. Fork the repository and create a feature branch
2. Make your changes
3. Run `uv run pytest` and `uv run ruff check .`
4. Submit a pull request with a properly formatted title and label

CI will run linting, type checking (pyright + mypy), tests across Python 3.10-3.13, PR title validation, and a build validation on every PR.
