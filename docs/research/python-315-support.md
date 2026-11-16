# Python 3.15 support validation

Research and implementation date: 2026-11-16. Scope: ordinary CPython 3.15 support for the SDK and CLI, retaining Python 3.12 as the minimum. This records local validation and the configured CI gates, not a release or deployment acceptance.

## Result

The complete locked dependency environment installs on Python 3.15.2 and Python 3.12.14 on macOS ARM64. The existing test suite passes on both interpreters: **4,229 passed, 1 skipped** each. No SDK runtime code or test behavior needed changing. [Package configuration](../../pyproject.toml), [lockfile](../../uv.lock), [test suite](../../tests/)

As of this date, Python 3.15.2 is the latest stable release, 3.15.0 was released on October 9, and 3.16.0a2 is a development preview. [Python downloads](https://www.python.org/downloads/), [3.15.2 release](https://www.python.org/downloads/release/python-3152/), [3.16.0a2 release](https://www.python.org/downloads/release/python-3160a2/)

## Dependency changes and observed failures

The first real Python 3.15 installation failed while compiling `fastuuid==0.14.0`, introduced by the previously locked LiteLLM 1.103.1. Its PyO3 0.26 build rejected Python 3.15 because its maximum supported version is 3.14. The upstream wheel request remains open. [fastuuid issue](https://github.com/fastuuid/fastuuid/issues/71)

LiteLLM 1.109.0 removes fastuuid from the Python 3.15 dependency graph and raises its Python 3.15 Pydantic and tiktoken requirements. All five owning extras now require LiteLLM `>=1.109.0,<2.0.0`, preserving the existing prerelease exclusion. The lock resolves 1.109.1. No PyO3 compatibility-check bypass or dependency fork was used. [LiteLLM 1.109.0 metadata](https://pypi.org/pypi/litellm/1.109.0/json), [owning extras](../../pyproject.toml)

Native dependency pins were refreshed for available Python 3.15 wheels. Representative tested versions are:

| Distribution | Previous lock | Validated lock |
| --- | --- | --- |
| Pydantic / pydantic-core | 2.13.5 / 2.46.5 | 2.14.3 / 2.50.4 |
| cffi | 2.0.0 | 2.1.1 |
| orjson | 3.11.9 | 3.13.0 |
| PyArrow | 24.0.0 | 26.0.1 |
| tiktoken | 0.12.0 | 0.14.0 |
| jiter | 0.12.0 | 0.19.0 |

These are tested candidates, not claims about the earliest compatible releases. The complete changes are in [uv.lock](../../uv.lock); LiteLLM also required boto3, botocore, and s3transfer updates. OpenAI 2.45.0, OpenAI Agents 0.20.0, Strands Agents 1.59.0, and Harbor 0.23.0 remained locked. Their existing upper bounds were retained, including OpenAI Agents `<0.21` and Harbor `<0.25`.

PyYAML 6.0.3, zstandard 0.25.0, PyIceberg 0.11.1, and watchdog 6.0.0 successfully built from source on Python 3.15.2. A full installation can therefore require a C compiler while upstream wheels remain unavailable. Source builds were serialized, and PyArrow source compilation was explicitly disabled. Earlier wheel-only probes had stopped at PyYAML; that was a distribution gap, not proof of runtime incompatibility. PyYAML's extension is optional, although other full-install dependencies require native compilation. [PyYAML source fallback](https://github.com/yaml/pyyaml/blob/6.0.3/setup.py#L244-L258)

## Validation evidence

All environments were isolated under `/private/tmp/osmosis-python315-validation`; the repository's existing `.venv` was not replaced. uv 0.13.8 performed installation because uv 0.13.4's embedded interpreter catalog did not include 3.15.2. The SDK's existing `uv>=0.13.4,<0.14` constraint and locked runtime uv 0.13.4 were retained.

| Check | Result |
| --- | --- |
| `uv sync --locked --all-extras --group dev`, Python 3.15.2 | Passed; 171 installed distributions |
| Same full environment, Python 3.12.14 | Passed; 172 installed distributions, including fastuuid on this older interpreter |
| `uv pip check`, both environments | Passed |
| Extra-module ownership check before the full suite | Passed |
| Complete pytest suite, Python 3.15.2 | 4,229 passed, 1 skipped, 47 warnings; 116.05 seconds |
| Complete pytest suite, Python 3.12.14 | 4,229 passed, 1 skipped, 47 warnings; 116.98 seconds |
| Ruff lint and format | Passed |
| `ty check --error-on-warning osmosis_ai/` | Passed |
| Pyright public API verification from a non-editable install | No unexpected errors; only the existing two Harbor external-base/metaclass errors exempted by CI |
| Existing isolated install verifier, Python 3.15 bare and full | Passed from outside the checkout with `python -I` and `PYTHONPATH` removed |

The isolated checks used non-editable installs of the checkout with locked dependencies. They verify installed package identity, metadata, CLI entry points, optional dependency boundaries, and feature imports through the existing [wheel verifier](../../.github/scripts/verify-wheel-install.py). CI separately builds the release artifacts and installs them with fresh dependency resolution.

## CI and compatibility boundaries

[tests.yml](../../.github/workflows/tests.yml) now runs the full test matrix on Python 3.12, 3.13, 3.14, and 3.15. Wheel installation checks retain all nine scenarios on 3.12 and add bare/full scenarios on 3.15. Coverage remains on 3.14. These are configured gates; consult the pull request checks for GitHub Actions acceptance.

The 3.15 classifier is added while `requires-python = ">=3.12"` and the Ruff, ty, and Pyright minimum targets remain at 3.12. Harbor's generated task environment still explicitly uses Python 3.12 in [tasks.py](../../osmosis_ai/rollout/backend/harbor/tasks.py); host-interpreter compatibility and sandbox Python are separate [runtime boundaries](../architecture.md#runtime-boundaries).

Python 3.15 enables UTF-8 mode by default. The passing local tests do not establish behavior for Windows legacy encodings or every external subprocess. [PEP 686](https://peps.python.org/pep-0686/) Free-threaded Python, new 3.15-only syntax, lazy imports, JIT tuning, and changing sandbox Python were outside this task.

## Remaining acceptance

Windows, local Python 3.13/3.14 regression runs, release-artifact production builds, cloud evaluation/training, and sandbox execution were not performed during local validation. Linux acceptance belongs to the configured CI gates. No release or deployment was made.
