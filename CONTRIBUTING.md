# Contributing

## Quick Start

**Using uv (recommended):**

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python
uv sync --all-extras --group dev
pre-commit install
uv run pytest
```

**Using pip:**

```bash
git clone https://github.com/Osmosis-AI/osmosis-sdk-python
cd osmosis-sdk-python
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]" --group dev
pre-commit install
pytest
```

## Commands Reference

The table below lists all development commands. If you installed with **pip**, drop the `uv run` prefix.

| Task | Command |
|------|---------|
| Run all tests | `uv run pytest` |
| Run a single file | `uv run pytest tests/unit/rollout/test_validator.py` |
| Run tests by name | `uv run pytest -k "test_name"` |
| Run with coverage | `uv run pytest --cov=osmosis_ai --cov-report=term-missing` |
| Lint | `uv run ruff check .` |
| Lint + autofix | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Check formatting | `uv run ruff format --check .` |
| Type check (pyright) | `uv run pyright osmosis_ai/` |
| Verify public API types | `uv run --no-editable pyright --verifytypes osmosis_ai --ignoreexternal` |

## Testing

Coverage configuration is in `pyproject.toml` under `[tool.coverage.*]`. CI enforces a minimum coverage threshold of 70%.

When changing remote run commands, preserve the naming convention: bare
verbs act on a group's primary noun. `train` and `eval` manage runs with
top-level `submit`, `list`, `info`, `logs`, and `stop` because the run is
their noun; `benchmark list|info` act on benchmarks themselves (workspace list and
benchmark page), with run lifecycle nested under `osmosis benchmark runs
list|info|logs|stop|download`. Eval and benchmark downloads
share the manifest transfer engine in `osmosis_ai/platform/cli/run_download.py`;
add domain-specific routes and fixed path classifiers instead of copying the
transfer loop.

## Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and code formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

Ruff is pinned to one version across `pyproject.toml`, `.pre-commit-config.yaml`, and CI so local checks, pre-commit hooks, and GitHub Actions produce the same results.

## Type Checking

[Pyright](https://microsoft.github.io/pyright/) is the type checker, included in
the `dev` dependency group.

- **Pyright** — must pass. All errors must be resolved before merging. It is
  installed from the `dev` dependency group.
- **Pyright `--verifytypes`** — must pass. Ensures all public API symbols have complete type annotations.

Configuration lives in `pyproject.toml` under `[tool.pyright]`.

> **Note:** `--verifytypes` requires a non-editable install. The `--no-editable` flag in `uv run` handles this automatically — it temporarily installs the package from a built wheel for the duration of the command.

## Pre-commit Hooks

[Pre-commit](https://pre-commit.com/) runs `ruff check --fix` and `ruff format` automatically on every commit. Make sure hooks are installed before submitting a pull request:

```bash
pre-commit install
```

> **Tip:** Run `uv run pyright osmosis_ai/` before pushing to catch type errors early. For full CI parity, also run `uv run --no-editable pyright --verifytypes osmosis_ai --ignoreexternal`. CI will block PRs with pyright failures.

## Pull Requests

### PR Title Format

All PR titles **must** follow this format (enforced by CI):

```
[module] type: description
[mod1][mod2] type: description          # multi-module (up to 3)
```

- **Modules**: `rollout`, `server`, `cli`, `auth`, `eval`, `misc`, `ci`, `doc`
- **Types**: `feat`, `fix`, `refactor`, `chore`, `test`, `doc`

For breaking changes, add `[BREAKING]` before the modules:

```
[BREAKING][module] type: description
```

For a stacked PR, prefix the whole title with its position in the stack:

```
[N/M][module] type: description
```

Use positive integers without leading zeroes, with `1 <= N <= M` and `M >= 2`.

A title prefix alone does not create a stack. Each PR must be based on the branch directly below it, with only `[1/M]` targeting `main`, and the PRs must be linked in GitHub from bottom to top. Use `gh stack submit` from the `github/gh-stack` extension when managing the stack locally, or link existing PRs without local tracking state:

```bash
gh stack link 291 292
```

**Examples:**

```
[rollout] feat: add streaming support for chat completions
[rollout][auth] refactor: extract LifecycleManager
[server] fix: handle timeout in rollout init
[cli] chore: update dependency versions
[BREAKING][rollout] refactor: change Grader.grade signature
[2/5][rollout] fix: surface harbor agent failures
```

PR titles appear directly in auto-generated GitHub Release Notes, so keep them clear and descriptive.

### Labels

**You do not need to add labels.** They are derived from the PR title and draft state and re-derived whenever the title changes, which is what keeps Release Notes categorized correctly:

| Label | Derived from |
|-------|--------------|
| `enhancement` | `feat:` |
| `bug` | `fix:` |
| `documentation` | `doc:` or `[doc]` |
| `refactor` | `refactor:` |
| `chore` | `chore:` or `test:` |
| `breaking` | a leading `[BREAKING]` |
| `stacked-pr` | a leading `[N/M]` |
| `wip` | the PR being a draft |
| `rollout` / `server` / `cli` / `auth` / `eval` / `ci` | the matching `[module]` bracket |

`[misc]` has no label of its own. `stacked-pr` and `wip` are filtering metadata and do not change the Release Notes category. Rollout work typically touches `osmosis_ai.rollout` and related CLI commands.

Labels you add yourself are never removed — priority and triage labels, `dependencies`, `reward`, and also any label the table above can produce. Adding `documentation` to a `[ci] fix:` PR that happens to touch the docs works and sticks; the workflow only reaps labels it applied itself. The flip side is that a label you add by hand is yours to remove by hand.

If the title does not match the convention, the labeling job fails rather than reconciling, so a malformed title cannot strip a PR's labels.

### Workflow

1. Fork the repository and create a feature branch
2. Make your changes
3. Run `uv run pytest` and `uv run ruff check .`
4. Submit a pull request with a properly formatted title; title-derived labels are added automatically

CI will run linting, type checking (pyright), tests on supported Python versions (see `requires-python` in `pyproject.toml`), PR title validation, and a build validation on every PR.
