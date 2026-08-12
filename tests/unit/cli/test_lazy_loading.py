"""Tests for lazy loading in osmosis_ai.__init__ and the CLI entry point.

Verifies that importing ``osmosis_ai`` does NOT eagerly pull in heavy
dependencies (litellm, openai, fastapi) and that rubric exports remain
accessible on demand. Rollout SDK types are not re-exported at package
top level — import from ``osmosis_ai.rollout`` directly.

CLI subprocess probes lock startup import discipline: ``--json`` must not
pull Rich, and ``--help`` must not pull optional/network stacks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

import osmosis_ai

# -- Subprocess isolation test ------------------------------------------------


def test_import_osmosis_ai_does_not_load_litellm():
    """Importing osmosis_ai in a fresh process must not eagerly load litellm."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import osmosis_ai; import sys; "
            "assert 'litellm' not in sys.modules, 'litellm was eagerly loaded'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"litellm was eagerly loaded: {result.stderr}"


# -- Rollout types not re-exported at top level --------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "AgentWorkflow",
        "Grader",
        "RolloutContext",
        "create_rollout_server",
    ],
)
def test_rollout_types_not_on_package(name: str):
    """Top-level osmosis_ai must not expose rollout SDK names."""
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(osmosis_ai, name)


def test_rollout_names_not_in_all():
    """__all__ must not list rollout SDK exports."""
    assert "AgentWorkflow" not in osmosis_ai.__all__
    assert "Grader" not in osmosis_ai.__all__


# -- Lazy rubric import -------------------------------------------------------


def test_lazy_rubric_import():
    """Accessing a rubric export via osmosis_ai triggers lazy import."""
    fn = osmosis_ai.evaluate_rubric
    assert callable(fn)


# -- Unknown attribute --------------------------------------------------------


def test_unknown_attribute_raises():
    """Accessing an undefined name raises AttributeError."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = osmosis_ai.this_does_not_exist  # type: ignore[attr-defined]


# -- __all__ completeness -----------------------------------------------------


def test_all_exports_accessible():
    """Every name listed in __all__ must be resolvable on the module."""
    missing: list[str] = []
    for name in osmosis_ai.__all__:
        try:
            getattr(osmosis_ai, name)
        except AttributeError:
            missing.append(name)
    assert missing == [], f"Names in __all__ that are not accessible: {missing}"


# -- Eager exports still present at module level ------------------------------


def test_eager_exports_present():
    """__version__ is available without lazy lookup."""
    assert isinstance(osmosis_ai.__version__, str)


# -- __getattr__ is defined ---------------------------------------------------


def test_module_has_getattr():
    """The module must define __getattr__ for rubric lazy loading."""
    assert hasattr(osmosis_ai, "__getattr__")
    assert callable(osmosis_ai.__getattr__)


# -- CLI startup import invariants (subprocess) --------------------------------


def _sys_modules_after_cli(args: list[str], *, cwd: str | None = None) -> set[str]:
    """Return ``sys.modules`` names after ``main(args)`` in a fresh process."""
    with tempfile.NamedTemporaryFile(
        prefix="osmosis-modules-", suffix=".json", delete=False
    ) as dump:
        dump_path = dump.name
    script = textwrap.dedent(
        """\
        import json
        import os
        import sys
        from osmosis_ai.cli.main import main

        try:
            main(sys.argv[1:])
        finally:
            with open(os.environ["OSMOSIS_MODULE_DUMP"], "w", encoding="utf-8") as fh:
                json.dump(list(sys.modules), fh)
        """
    )
    env = os.environ.copy()
    env["OSMOSIS_MODULE_DUMP"] = dump_path
    try:
        subprocess.run(
            [sys.executable, "-c", script, *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
        )
        with open(dump_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        return set(loaded)
    finally:
        os.unlink(dump_path)


def _top_level_modules(loaded: set[str]) -> set[str]:
    return {name.partition(".")[0] for name in loaded}


def test_json_cli_does_not_load_rich_stack() -> None:
    """``osmosis --json dataset list`` must not import rich / pygments / markdown_it.

    Runs a real command in a scratch cwd so the handler imports ``cli.console``
    (via ``platform.cli.dataset``) and then fails closed on workspace
    validation — no login or network required. ``cli.console`` must not pull
    the Rich stack just because it was imported.
    """
    with tempfile.TemporaryDirectory() as tmp:
        loaded = _sys_modules_after_cli(["--json", "dataset", "list"], cwd=tmp)
    assert "osmosis_ai.platform.cli.dataset" in loaded
    assert "osmosis_ai.cli.console" in loaded
    roots = _top_level_modules(loaded)
    leaked = {"rich", "pygments", "markdown_it"} & roots
    assert not leaked, f"JSON CLI loaded UI stack: {sorted(leaked)}"


def test_help_does_not_load_heavy_optional_deps() -> None:
    """``osmosis --help`` must not import httpx, keyring, urllib.request, litellm, fastapi."""
    loaded = _sys_modules_after_cli(["--help"])
    roots = _top_level_modules(loaded)
    leaked_roots = {"httpx", "keyring", "litellm", "fastapi"} & roots
    assert not leaked_roots, f"--help loaded optional deps: {sorted(leaked_roots)}"
    assert "urllib.request" not in loaded, "urllib.request was loaded during --help"


def test_output_package_import_does_not_load_api_models() -> None:
    """Importing an output submodule must not run serializers → platform.api.models."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from osmosis_ai.cli.output.context import OutputContext; import sys; "
            "assert 'osmosis_ai.cli.output.serializers' not in sys.modules; "
            "assert 'osmosis_ai.platform.api.models' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
