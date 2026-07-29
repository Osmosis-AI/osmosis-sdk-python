#!/usr/bin/env python3
"""Verify an isolated installation of an osmosis-ai wheel.

CI executes this from a temporary directory with ``python -I`` and
``PYTHONPATH`` removed so a source checkout cannot satisfy any imports. Apart
from ``packaging`` (a declared base dependency used to inspect wheel metadata),
the verifier uses only the standard library.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement

BASE_REQUIREMENTS = {
    "cryptography",
    "httpx",
    "keyring",
    "packaging",
    "prompt-toolkit",
    "pydantic",
    "python-dotenv",
    "questionary",
    "requests",
    "rich",
    "typer",
}

EXTRA_REQUIREMENTS: dict[str, set[str]] = {
    "server": {"click", "fastapi", "uvicorn"},
    "strands": {
        "aiohttp",
        "click",
        "litellm",
        "mcp",
        "orjson",
        "strands-agents",
    },
    "openai-agents": {
        "aiohttp",
        "click",
        "litellm",
        "mcp",
        "openai-agents",
        "orjson",
    },
    "harbor": {
        "aiohttp",
        "click",
        "dockerfile-parse",
        "harbor",
        "litellm",
        "orjson",
        "toml",
    },
    "rubric": {"aiohttp", "click", "litellm", "orjson", "tqdm"},
    "parquet": {"pyarrow"},
    "full": {"osmosis-ai"},
}


def _normalize_distribution(name: str) -> str:
    """Return the normalized project name used for metadata comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions() -> set[str]:
    return {
        _normalize_distribution(distribution.metadata["Name"])
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }


def _assert_distributions(
    installed: set[str],
    *,
    present: Iterable[str] = (),
    absent: Iterable[str] = (),
) -> None:
    expected = {_normalize_distribution(name) for name in present}
    forbidden = {_normalize_distribution(name) for name in absent}
    missing = sorted(expected - installed)
    unexpected = sorted(forbidden & installed)
    assert not missing, f"Expected distributions are missing: {', '.join(missing)}"
    assert not unexpected, "Unselected distributions were installed: " + ", ".join(
        unexpected
    )


def _assert_dependency_metadata(
    distribution: importlib.metadata.Distribution,
) -> None:
    """Verify that every direct dependency belongs to its intended feature."""
    provided_extras = {
        _normalize_distribution(extra)
        for extra in distribution.metadata.get_all("Provides-Extra", [])
    }
    assert provided_extras == set(EXTRA_REQUIREMENTS), (
        "Unexpected Provides-Extra metadata: "
        f"expected={sorted(EXTRA_REQUIREMENTS)}, actual={sorted(provided_extras)}"
    )

    actual_base: set[str] = set()
    actual_extras = {extra: set() for extra in EXTRA_REQUIREMENTS}
    for raw_requirement in distribution.requires or []:
        requirement = Requirement(raw_requirement)
        name = _normalize_distribution(requirement.name)
        if requirement.marker is None:
            actual_base.add(name)
            continue

        matched_extras: list[str] = []
        for extra in EXTRA_REQUIREMENTS:
            environment = default_environment()
            environment["extra"] = extra
            if requirement.marker.evaluate(environment):
                actual_extras[extra].add(name)
                matched_extras.append(extra)
        assert matched_extras, (
            "A marked requirement is not owned by any declared extra: "
            f"{raw_requirement}"
        )

    assert actual_base == BASE_REQUIREMENTS, (
        "Unexpected base dependency metadata: "
        f"expected={sorted(BASE_REQUIREMENTS)}, actual={sorted(actual_base)}"
    )
    assert actual_extras == EXTRA_REQUIREMENTS, (
        "Unexpected extra dependency metadata: "
        f"expected={EXTRA_REQUIREMENTS}, actual={actual_extras}"
    )


def _assert_public_exports(module_name: str, symbols: Iterable[str]) -> None:
    module = importlib.import_module(module_name)
    exported = set(getattr(module, "__all__", ()))
    for symbol in symbols:
        value = getattr(module, symbol)
        assert value is not None, (
            f"{module_name}.{symbol} unexpectedly resolved to None"
        )
        assert symbol in exported, f"{module_name}.{symbol} is missing from __all__"


def _smoke_bare() -> None:
    osmosis_ai = importlib.import_module("osmosis_ai")
    assert "evaluate_rubric" not in osmosis_ai.__all__
    star_namespace: dict[str, Any] = {}
    exec("from osmosis_ai import *", star_namespace)
    assert "RubricResult" in star_namespace
    assert "evaluate_rubric" not in star_namespace

    _assert_public_exports(
        "osmosis_ai.rollout",
        (
            "AgentWorkflow",
            "ExecutionBackend",
            "Grader",
            "LocalBackend",
            "SampleSource",
        ),
    )


def _smoke_server() -> None:
    _assert_public_exports(
        "osmosis_ai.rollout.server",
        ("ControllerAuth", "create_rollout_server"),
    )


def _smoke_strands() -> None:
    _assert_public_exports(
        "osmosis_ai.rollout.integrations.strands",
        ("OsmosisRolloutModel", "OsmosisStrandsAgent"),
    )


def _smoke_openai_agents() -> None:
    _assert_public_exports(
        "osmosis_ai.rollout.integrations.openai_agents",
        (
            "OsmosisAgent",
            "OsmosisLitellmModel",
            "OsmosisMemorySession",
            "OsmosisRolloutModel",
            "SessionSampleSource",
        ),
    )


def _smoke_harbor() -> None:
    _assert_public_exports(
        "osmosis_ai.rollout.backend.harbor",
        ("HarborBackend", "OsmosisInstalledAgent"),
    )


def _smoke_rubric() -> None:
    _assert_public_exports(
        "osmosis_ai.eval.rubric",
        (
            "MissingAPIKeyError",
            "ModelNotFoundError",
            "ProviderRequestError",
            "RubricResult",
            "evaluate_rubric",
        ),
    )


def _smoke_parquet() -> None:
    importlib.import_module("pyarrow.parquet")
    dataset = importlib.import_module("osmosis_ai.platform.cli.dataset")
    assert callable(dataset.validate)


SCENARIO_SMOKE: dict[str, Callable[[], None]] = {
    "bare": _smoke_bare,
    "server": _smoke_server,
    "strands": _smoke_strands,
    "openai-agents": _smoke_openai_agents,
    "harbor": _smoke_harbor,
    "rubric": _smoke_rubric,
    "parquet": _smoke_parquet,
}

SCENARIO_PRESENT: dict[str, set[str]] = {
    "bare": set(),
    "server": {"fastapi", "uvicorn"},
    "strands": {"litellm", "strands-agents"},
    "openai-agents": {"litellm", "openai-agents"},
    "harbor": {"dockerfile-parse", "harbor", "toml"},
    "rubric": {"litellm", "orjson", "tqdm"},
    "parquet": {"pyarrow"},
    "full": {
        "fastapi",
        "dockerfile-parse",
        "harbor",
        "litellm",
        "openai-agents",
        "orjson",
        "pyarrow",
        "strands-agents",
        "toml",
        "tqdm",
        "uvicorn",
    },
}

# These are feature-owner distributions, not incidental transitive packages.
# Harbor itself currently depends on FastAPI and LiteLLM, for example, so those
# cannot be used to infer whether the server/rubric extras were selected there.
SCENARIO_ABSENT: dict[str, set[str]] = {
    "bare": {
        "dockerfile-parse",
        "fastapi",
        "harbor",
        "litellm",
        "openai-agents",
        "orjson",
        "pyarrow",
        "strands-agents",
        "tqdm",
        "toml",
        "uvicorn",
    },
    "server": {
        "dockerfile-parse",
        "harbor",
        "litellm",
        "openai-agents",
        "orjson",
        "pyarrow",
        "strands-agents",
        "toml",
        "tqdm",
    },
    "strands": {"dockerfile-parse", "harbor", "openai-agents", "pyarrow", "toml"},
    "openai-agents": {
        "dockerfile-parse",
        "harbor",
        "pyarrow",
        "strands-agents",
        "toml",
    },
    "harbor": {"openai-agents", "pyarrow", "strands-agents"},
    "rubric": {
        "dockerfile-parse",
        "fastapi",
        "harbor",
        "openai-agents",
        "pyarrow",
        "strands-agents",
        "toml",
        "uvicorn",
    },
    "parquet": {
        "dockerfile-parse",
        "fastapi",
        "harbor",
        "litellm",
        "openai-agents",
        "orjson",
        "strands-agents",
        "toml",
        "tqdm",
        "uvicorn",
    },
    "full": set(),
}

# Sandbox runtimes are supplied by the remote rollout environment. The wheel
# must never pull retired Daytona or either SkyPilot distribution.
PROHIBITED_SANDBOX_DISTRIBUTIONS = {"daytona", "skypilot", "skypilot-nightly"}


def _assert_clean_import_state() -> None:
    """Ensure importing rollout core did not initialize optional/CLI modules."""
    forbidden_prefixes = (
        "agents",
        "dotenv",
        "fastapi",
        "harbor",
        "keyring",
        "litellm",
        "prompt_toolkit",
        "pyarrow",
        "questionary",
        "rich",
        "strands",
        "typer",
        "uvicorn",
        "osmosis_ai.cli",
        "osmosis_ai.eval.rubric",
        "osmosis_ai.platform",
        "osmosis_ai.rollout.backend.harbor",
        "osmosis_ai.rollout.integrations",
        "osmosis_ai.rollout.server",
    )
    loaded = sorted(
        module_name
        for module_name in sys.modules
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert not loaded, "Rollout core eagerly loaded optional/CLI modules: " + ", ".join(
        loaded
    )


def _assert_wheel_identity(
    source_root: Path,
) -> tuple[Any, importlib.metadata.Distribution]:
    assert sys.flags.isolated == 1, "Smoke verification must run with python -I"
    assert "PYTHONPATH" not in os.environ, "PYTHONPATH must be removed for wheel smoke"

    cwd = Path.cwd().resolve()
    source_root = source_root.resolve()
    assert cwd != source_root and source_root not in cwd.parents, (
        f"Smoke verification must run outside the source checkout (cwd={cwd})"
    )

    osmosis_ai = importlib.import_module("osmosis_ai")
    distribution = importlib.metadata.distribution("osmosis-ai")
    module_path = Path(osmosis_ai.__file__).resolve()
    metadata_path = Path(distribution.locate_file("osmosis_ai/__init__.py")).resolve()

    assert module_path == metadata_path, (
        "Imported osmosis_ai does not belong to the installed distribution: "
        f"module={module_path}, metadata={metadata_path}"
    )
    assert source_root not in module_path.parents, (
        f"Imported osmosis_ai from the source checkout: {module_path}"
    )
    assert osmosis_ai.__version__ == distribution.version
    return osmosis_ai, distribution


def _assert_cli_aliases(version: str) -> None:
    expected = f"osmosis-ai {version}"
    for alias in ("osmosis", "osmosis_ai", "osmosis-ai"):
        executable = Path(sys.executable).parent / alias
        assert executable.is_file(), f"Missing console script: {executable}"
        result = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() == expected, (
            f"Unexpected `{alias} --version` output: {result.stdout!r}"
        )
        assert not result.stderr, (
            f"Unexpected `{alias} --version` stderr: {result.stderr!r}"
        )

    # Command registration touches every top-level CLI group. This catches a
    # missing optional dependency that a short-circuited --version path would
    # otherwise miss while keeping the three alias checks inexpensive.
    help_result = subprocess.run(
        [str(Path(sys.executable).parent / "osmosis"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Usage: osmosis" in help_result.stdout
    assert "Osmosis AI CLI." in help_result.stdout
    assert not help_result.stderr, (
        f"Unexpected `osmosis --help` stderr: {help_result.stderr!r}"
    )

    rubric_result = subprocess.run(
        [
            str(Path(sys.executable).parent / "osmosis"),
            "eval",
            "rubric",
            "--data",
            "/does-not-exist.jsonl",
            "--rubric",
            "score correctness",
            "--model",
            "openai/example",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rubric_result.returncode == 1
    assert 'pip install "osmosis-ai[rubric]"' in rubric_result.stderr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=(*SCENARIO_SMOKE, "full"))
    parser.add_argument("--source-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _, distribution = _assert_wheel_identity(args.source_root)
    _assert_dependency_metadata(distribution)

    importlib.import_module("osmosis_ai.rollout")
    _assert_clean_import_state()

    installed = _installed_distributions()
    present = SCENARIO_PRESENT.get(args.scenario, set())
    absent = (
        SCENARIO_ABSENT.get(args.scenario, set()) | PROHIBITED_SANDBOX_DISTRIBUTIONS
    )
    _assert_distributions(
        installed,
        present={"osmosis-ai", *present},
        absent=absent,
    )

    if args.scenario == "full":
        for smoke in SCENARIO_SMOKE.values():
            smoke()
    else:
        SCENARIO_SMOKE[args.scenario]()

    if args.scenario == "bare":
        _assert_cli_aliases(distribution.version)

    print(
        f"Verified osmosis-ai {distribution.version} "
        f"({args.scenario}, {len(installed)} distributions)"
    )


if __name__ == "__main__":
    main()
