"""Tests for lazy loading in osmosis_ai.__init__.

Verifies that importing ``osmosis_ai`` does NOT eagerly pull in heavy
dependencies (litellm, openai, fastapi) and that rubric exports remain
accessible on demand. Rollout SDK types are not re-exported at package
top level — import from ``osmosis_ai.rollout`` directly.
"""

from __future__ import annotations

import subprocess
import sys

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
