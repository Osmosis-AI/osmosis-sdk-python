# Contributing

## Setup

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python

# Install with all development dependencies
pip install -e ".[dev]"

# Or using uv
uv sync --all-extras

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

CI will run linting (`ruff check` + `ruff format --check`) and tests with coverage on every PR.
