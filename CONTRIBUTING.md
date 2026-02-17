# Contributing

## Setup

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python

# Install with all development dependencies (includes server + mcp extras
# for type checking and full test coverage)
pip install -e ".[dev]"

# Or using uv (recommended)
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

## Testing

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/unit/rollout/core/test_base.py

# Run tests matching a pattern
pytest -k "test_name"

# Run with coverage report
pytest --cov=osmosis_ai --cov-report=term-missing
```

Coverage configuration is in `pyproject.toml` under `[tool.coverage.*]`. CI enforces a minimum coverage threshold.

## Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
# Check for lint errors
ruff check .

# Auto-fix lint errors where possible
ruff check --fix .

# Format code
ruff format .

# Check formatting without modifying files
ruff format --check .
```

Ruff is pinned to one version across `pyproject.toml`, `.pre-commit-config.yaml`, and CI so local checks, pre-commit hooks, and GitHub Actions produce the same results.

## Type Checking

This project uses [Pyright](https://microsoft.github.io/pyright/) as the primary type checker and [mypy](https://mypy-lang.org/) as a secondary checker. Both are included in the `dev` extras — install with:

```bash
uv sync --extra dev
```

Then run via `uv run`:

```bash
# Pyright — must pass (blocking in CI)
uv run pyright osmosis_ai/

# mypy — advisory (non-blocking in CI)
uv run mypy osmosis_ai/
```

Pyright runs in `standard` mode. All pyright errors must be resolved before merging. mypy runs in CI with `continue-on-error` and serves as an advisory check — fix mypy warnings when practical, but they won't block a PR.

Configuration for both tools lives in `pyproject.toml` under `[tool.pyright]` and `[tool.mypy]`.

> **Note:** CI also runs `pyright --verifytypes osmosis_ai --ignoreexternal` to check public API type completeness, but this command requires a non-editable install to locate the `py.typed` marker. It does not work with local editable installs (`uv sync`) and is therefore non-blocking in CI.

## Pre-commit Hooks

A [pre-commit](https://pre-commit.com/) configuration is included to run Ruff automatically on every commit:

```bash
pre-commit install
```

This ensures `ruff check --fix` and `ruff format` run before each commit. Please make sure hooks are installed before submitting a pull request.

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

CI will run linting (`ruff check` + `ruff format --check`), type checking (pyright + mypy), and tests with coverage on every PR.
